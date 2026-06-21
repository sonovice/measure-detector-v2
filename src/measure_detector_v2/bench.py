from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from time import perf_counter

from measure_detector_v2.detector import MeasureDetector, default_model_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark direct detector inference.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    detector = MeasureDetector(args.model, conf=args.conf)
    data = args.image.read_bytes()
    for _ in range(args.warmup):
        detector.predict_bytes(data)

    times = []
    n_measures = 0
    for _ in range(args.runs):
        start = perf_counter()
        measures, _, _ = detector.predict_bytes(data)
        times.append((perf_counter() - start) * 1000)
        n_measures = len(measures)

    times_sorted = sorted(times)
    p95 = times_sorted[int(round(0.95 * (len(times_sorted) - 1)))]
    print(f"runs={args.runs}")
    print(f"measures={n_measures}")
    print(f"mean_ms={statistics.mean(times):.1f}")
    print(f"median_ms={statistics.median(times):.1f}")
    print(f"p95_ms={p95:.1f}")
    print(f"min_ms={min(times):.1f}")
    print(f"max_ms={max(times):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
