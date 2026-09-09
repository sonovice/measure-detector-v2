import unittest
import xml.etree.ElementTree as ET
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from measure_detector_v2.inputs import decode_pages
from measure_detector_v2.postprocess import BBox
from measure_detector_v2.server import app
from measure_detector_v2.xfdf import PageGeometry, pdf_geometry, write_xfdf
from test_pdf import FakeDetector, PDF

NS = {'x': 'http://ns.adobe.com/xfdf/', 'h': 'http://www.w3.org/1999/xhtml'}


class XfdfTests(unittest.TestCase):
    def setUp(self):
        app.state.detector = FakeDetector()
        self.client = TestClient(app)

    def test_endpoint_annotation_structure_and_numbering(self):
        response = self.client.post('/xfdf', files={'files': ('score & parts.pdf', PDF)}, data={'pretty': 'y'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['content-type'], 'application/vnd.adobe.xfdf')
        root = ET.fromstring(response.content)
        self.assertEqual(root.find('x:f', NS).get('href'), 'score & parts.pdf')
        annotations = root.findall('x:annots/x:freetext', NS)
        self.assertEqual([a.get('page') for a in annotations], ['0', '1'])
        self.assertEqual([a.get('subject') for a in annotations], ['1', '2'])
        self.assertEqual([a.get('name') for a in annotations], ['bar-0', 'bar-1'])
        self.assertEqual(annotations[0].get('rect'), '14.400000,7.200000,115.200000,57.600000')
        for i, a in enumerate(annotations, 1):
            self.assertEqual(a.get('title'), 'BarNumber')
            self.assertEqual(a.get('opacity'), '0.1')
            self.assertEqual(a.find('x:contents', NS).text, str(i))
            self.assertEqual(a.find('x:contents-richtext/h:body/h:p', NS).text, str(i))
            self.assertIn('/Courier#20New', a.find('x:defaultappearance', NS).text)

    def test_accepts_only_one_pdf(self):
        for files in [[('files', ('image.png', b'not a PDF'))],
                      [('files', ('one.pdf', PDF)), ('files', ('two.pdf', PDF))]]:
            response = self.client.post('/xfdf', files=files)
            self.assertEqual(response.status_code, 400)

    def test_missing_poppler_and_no_detections(self):
        with patch('measure_detector_v2.inputs.subprocess.run', side_effect=FileNotFoundError):
            self.assertEqual(self.client.post('/xfdf', files={'files': ('score.pdf', PDF)}).status_code, 503)
        root = ET.fromstring(write_xfdf('blank.pdf', [(PageGeometry((0, 0, 100, 100), 0), [])]))
        self.assertEqual(len(root.find('x:annots', NS)), 0)

    def test_rotated_rendered_boxes_map_back_to_pdf_coordinates(self):
        data = (Path(__file__).parent / 'fixtures/rotated-offset.pdf').read_bytes()
        geometries = pdf_geometry(data)
        self.assertEqual([g.rotation for g in geometries], [0, 90, 180, 270])
        with closing(decode_pages(data)) as pages:
            for page, rgb in pages:
                ys, xs = np.where((rgb[:, :, 0] > 240) & (rgb[:, :, 1] < 10))
                height, width = rgb.shape[:2]
                bbox = BBox(xs.min() / width, ys.min() / height,
                            (xs.max() + 1) / width, (ys.max() + 1) / height)
                rect = geometries[page - 1].rect(bbox)
                # Rendering rounds to pixels; export should recover the source rectangle.
                np.testing.assert_allclose(rect, [50, 40, 150, 80], atol=0.8)

    def test_rotation_transform_exact(self):
        bbox = BBox(0.1, 0.2, 0.8, 0.9)
        expected = [(30, 30, 170, 100), (50, 30, 190, 100),
                    (50, 40, 190, 110), (30, 40, 170, 110)]
        for rotation, rect in zip([0, 90, 180, 270], expected):
            np.testing.assert_allclose(PageGeometry((10, 20, 210, 120), rotation).rect(bbox), rect)
