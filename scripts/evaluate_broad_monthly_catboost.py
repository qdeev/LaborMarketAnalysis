"""Evaluate broad monthly CatBoost predictions."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import BroadMonthlyModelEvaluator  # noqa: E402
from src.preprocessing import PROCESSED_DATA_DIR  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROCESSED_DATA_DIR / "predictions_broad_monthly_test.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_evaluation.json",
    )
    parser.add_argument(
        "--group-metrics",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_group_metrics.csv",
    )
    args = parser.parse_args()

    report = BroadMonthlyModelEvaluator().evaluate_file(
        predictions_path=args.predictions,
        output_path=args.output,
        group_metrics_path=args.group_metrics,
    )
    overall = report["overall"]
    print(
        "broad monthly evaluation: "
        f"rows={report['row_count']}, "
        f"catboost_mape={overall['catboost']['mape']:.2f}%, "
        f"baseline_mape={overall['baseline']['mape']:.2f}%, "
        f"output={report['output_path']}, "
        f"group_metrics={report['group_metrics_path']}"
    )


if __name__ == "__main__":
    main()
