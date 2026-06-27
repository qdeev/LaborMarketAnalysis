"""Prepare broad supervised next-observed-quarter salary ML dataset."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import BroadQuarterlySalaryMLDatasetBuilder, PROCESSED_DATA_DIR  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "ml_broad_quarterly_salary_dataset.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROCESSED_DATA_DIR / "ml_broad_quarterly_salary_dataset_report.json",
    )
    args = parser.parse_args()

    result = BroadQuarterlySalaryMLDatasetBuilder().prepare_file(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
    )
    print(
        "broad quarterly salary ML dataset: "
        f"rows={len(result.dataframe)}, "
        f"output={result.output_path}, "
        f"report={result.report_path}"
    )


if __name__ == "__main__":
    main()
