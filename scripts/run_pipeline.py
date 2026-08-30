"""Run the drone tamper-detection experiment end to end (Steps 1-7).

Usage:
    uv run python scripts/run_pipeline.py                  # everything
    uv run python scripts/run_pipeline.py --skip rbf,rbf_lc   # steps 1-4 and 5 only
    uv run python scripts/run_pipeline.py --only main         # just the Step 1-4 block

Stages: main (Steps 1-4, replicate 0), robustness (Step 5, replicates 0-3),
rbf (Step 6, replicates 0-3, needs robustness's results/robustness_row_rep00_03.csv),
rbf_lc (Step 7, replicate 0).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import sklearn

from drone_tamper import experiment
from drone_tamper.config import ROOT, ensure_dirs

STAGES = ["main", "robustness", "rbf", "rbf_lc"]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_stages(value: str) -> list[str]:
    stages = [s.strip() for s in value.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stage(s) {sorted(unknown)}; choose from {STAGES}")
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=parse_stages, default=None, help="comma-separated stages to run")
    parser.add_argument("--skip", type=parse_stages, default=None, help="comma-separated stages to skip")
    args = parser.parse_args()

    stages = args.only if args.only is not None else STAGES
    if args.skip:
        stages = [s for s in stages if s not in args.skip]

    ensure_dirs()
    log_path = ROOT / "results" / "pipeline_run.log"
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    try:
        print("python", sys.version.split()[0])
        print("pandas", pd.__version__)
        print("numpy", np.__version__)
        print("scikit-learn", sklearn.__version__)
        print("stages:", stages)

        if "main" in stages:
            experiment.run_main(replicate=0)
        if "robustness" in stages:
            experiment.run_robustness()
        if "rbf" in stages:
            experiment.run_rbf()
        if "rbf_lc" in stages:
            experiment.run_rbf_learning_curve(replicate=0)

        print(f"\nDone. Full log written to {log_path}")
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    main()
