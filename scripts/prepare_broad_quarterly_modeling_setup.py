"""Prepare feature, split, and baseline report for broad quarterly salary modeling."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import BroadQuarterlyModelingSetup, PROCESSED_DATA_DIR  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "ml_broad_quarterly_salary_dataset.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "broad_quarterly_modeling_setup.json",
    )
    args = parser.parse_args()

    report = BroadQuarterlyModelingSetup().prepare_file(
        input_path=args.input,
        output_path=args.output,
    )
    split = report["split"]
    test_mape = report["baseline_metrics"]["test"].get("mape")
    print(
        "broad quarterly modeling setup: "
        f"rows={report['dataset_row_count']}, "
        f"train={split['train']['row_count']}, "
        f"validation={split['validation']['row_count']}, "
        f"test={split['test']['row_count']}, "
        f"baseline_test_mape={test_mape:.2f}%, "
        f"output={report['output_path']}"
    )


if __name__ == "__main__":
    main()
