"""Aggregate canonical vacancies to monthly segment parquet."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PROCESSED_DATA_DIR, SegmentAggregator  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "canonical_vacancies.parquet",
        help="Input canonical vacancies parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
        help="Output monthly segments parquet.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    segments = SegmentAggregator().save_monthly_segments(df, output_path=args.output)
    print(f"monthly_segments rows={len(segments)}, output={args.output}")


if __name__ == "__main__":
    main()
