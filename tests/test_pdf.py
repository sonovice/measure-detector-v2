import io
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from measure_detector_v2.inputs import decode_pages
from measure_detector_v2.postprocess import BBox, Measure
from measure_detector_v2.server import app

PDF = (Path(__file__).parent / 'fixtures/two-pages.pdf').read_bytes()


class FakeDetector:
    def predict_image(self, rgb, **options):
        return [Measure(1, 'typeset', 0.9, BBox(0.1, 0.2, 0.8, 0.9))], 'typeset', 0.9


class PdfTests(unittest.TestCase):
    def setUp(self):
        app.state.detector = FakeDetector()
        self.client = TestClient(app)

    def test_render_order_size_and_rgb(self):
        pages = list(decode_pages(PDF))
        self.assertEqual([p for p, _ in pages], [1, 2])
        self.assertEqual([rgb.shape for _, rgb in pages], [(150, 300, 3), (300, 150, 3)])
        self.assertEqual(pages[0][1][75, 75].tolist(), [255, 0, 0])
        self.assertEqual(pages[1][1][225, 75].tolist(), [0, 0, 255])
        self.assertEqual(pages[0][1][75, 225].tolist(), [255, 255, 255])

    def test_json_mixed_inputs_single_response(self):
        png = cv2.imencode('.png', np.zeros((10, 20, 3), dtype=np.uint8))[1].tobytes()
        response = self.client.post('/json', files=[
            ('files', ('score.PDF', PDF, 'application/octet-stream')),
            ('files', ('page.png', png, 'image/png')),
        ], data={'pretty': 'y', 'auto': 'y'})
        self.assertEqual(response.status_code, 200)
        results = response.json()['results']
        self.assertEqual([r.get('page') for r in results], [1, 2, None])
        self.assertEqual([r['filename'] for r in results], ['score.PDF', 'score.PDF', 'page.png'])
        self.assertEqual(results[0]['measures'][0]['bbox']['x1'], 0.1)

    def test_mei_pages_and_pixel_coordinates(self):
        response = self.client.post('/mei', files={'files': ('score.pdf', PDF)})
        self.assertEqual(response.status_code, 200)
        root = ET.fromstring(response.content)
        ns = {'m': 'http://www.music-encoding.org/ns/mei'}
        graphics = root.findall('.//m:graphic', ns)
        self.assertEqual([g.get('target') for g in graphics], ['score.pdf#page=1', 'score.pdf#page=2'])
        self.assertEqual([g.get('width') for g in graphics], ['300px', '150px'])
        self.assertEqual([z.get('ulx') for z in root.findall('.//m:zone', ns)], ['30', '15'])
        self.assertEqual([m.get('n') for m in root.findall('.//m:measure', ns)], ['1', '2'])

    def test_debug_selected_page_and_range(self):
        response = self.client.post('/debug', files={'file': ('score.pdf', PDF)}, data={'page': 2})
        self.assertEqual(response.status_code, 200)
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (300, 150))
        for page, status in [(0, 422), (3, 400)]:
            response = self.client.post('/debug', files={'file': ('score.pdf', PDF)}, data={'page': page})
            self.assertEqual(response.status_code, status)

    def test_invalid_pdf_and_missing_poppler(self):
        for endpoint, field in [('/json', 'files'), ('/mei', 'files'), ('/debug', 'file')]:
            response = self.client.post(endpoint, files={field: ('bad.pdf', b'%PDF-broken')})
            self.assertEqual(response.status_code, 400)
        with patch('measure_detector_v2.inputs.subprocess.run', side_effect=FileNotFoundError):
            response = self.client.post('/json', files={'files': ('score.pdf', PDF)})
        self.assertEqual(response.status_code, 503)


if __name__ == '__main__':
    unittest.main()
