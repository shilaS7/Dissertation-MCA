"""Extract the profile CSVs needed by the experiment from the Kaggle archive."""

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile


PROFILES = ("balanced", "strong", "subtle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("drone.zip"))
    parser.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    archive = args.archive if args.archive.is_absolute() else root / args.archive

    with ZipFile(archive) as source:
        for replicate in args.replicates:
            for profile in PROFILES:
                member = (
                    f"drone_temparing_dataset_v2/{profile}/rep_{replicate:02d}/"
                    "tampering_research_dataset.csv"
                )
                destination = root / "data" / "raw" / f"rep_{replicate:02d}" / f"{profile}.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_file, destination.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file)
                print(f"extracted {member} -> {destination}")


if __name__ == "__main__":
    main()
