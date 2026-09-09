use crate::Measure;

// A pairwise "same system" comparison is not transitive and can make Rust's
// sort panic. Assign boxes to systems first, then use total coordinate orders.
pub(crate) fn sort(measures: &mut [Measure]) {
    let mut ordered = measures.to_vec();
    ordered.sort_by(|a, b| {
        a.bbox
            .y1
            .total_cmp(&b.bbox.y1)
            .then_with(|| a.bbox.x1.total_cmp(&b.bbox.x1))
    });
    let mut systems: Vec<Vec<Measure>> = Vec::new();
    for measure in ordered {
        let system = systems.iter().position(|system| {
            // Keep the first (topmost) box as a fixed anchor; overlapping boxes
            // must not gradually merge two separate systems via a chain.
            let anchor = &system[0].bbox;
            let bbox = &measure.bbox;
            let height = (anchor.y2 - anchor.y1).min(bbox.y2 - bbox.y1);
            let overlap = anchor.y2.min(bbox.y2) - anchor.y1.max(bbox.y1);
            height > 0.0 && overlap / height >= 0.5
        });
        match system {
            Some(index) => systems[index].push(measure),
            None => systems.push(vec![measure]),
        }
    }
    for system in &mut systems {
        system.sort_by(|a, b| {
            a.bbox
                .x1
                .total_cmp(&b.bbox.x1)
                .then_with(|| a.bbox.y1.total_cmp(&b.bbox.y1))
        });
    }
    for (destination, measure) in measures.iter_mut().zip(systems.into_iter().flatten()) {
        *destination = measure;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BBox;

    fn measure(index: usize, [x1, y1, x2, y2]: [f32; 4]) -> Measure {
        Measure {
            class_id: index,
            class_name: "typeset".into(),
            confidence: 0.9,
            bbox: BBox { x1, y1, x2, y2 },
        }
    }

    #[test]
    fn real_page_97_detections_do_not_panic_or_lose_boxes() {
        let boxes: Vec<[f32; 4]> =
            serde_json::from_str(include_str!("../tests/page-97-boxes.json")).unwrap();
        let mut measures: Vec<_> = boxes
            .into_iter()
            .enumerate()
            .map(|(i, b)| measure(i, b))
            .collect();
        sort(&mut measures);
        let mut ids: Vec<_> = measures.iter().map(|m| m.class_id).collect();
        ids.sort();
        assert_eq!(ids, (0..measures.len()).collect::<Vec<_>>());
    }

    #[test]
    fn staggered_boxes_follow_systems_then_horizontal_position() {
        let mut measures = vec![
            measure(0, [0.1, 0.5, 0.3, 0.7]),
            measure(1, [0.5, 0.1, 0.8, 0.3]),
            measure(2, [0.1, 0.12, 0.3, 0.31]),
            measure(3, [0.5, 0.51, 0.8, 0.71]),
        ];
        sort(&mut measures);
        assert_eq!(
            measures.iter().map(|m| m.class_id).collect::<Vec<_>>(),
            vec![2, 1, 0, 3]
        );
    }
}
