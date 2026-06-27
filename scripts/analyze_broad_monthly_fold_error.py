"""Analyze errors for one broad monthly CatBoost backtesting fold."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import BroadMonthlyFoldErrorAnalyzer  # noqa: E402
from src.preprocessing import PROCESSED_DATA_DIR  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet",
    )
    parser.add_argument(
        "--setup",
        type=Path,
        default=PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json",
    )
    parser.add_argument(
        "--best-params",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_best_params.json",
    )
    parser.add_argument(
        "--backtest-metrics",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_2025_12_error_analysis.json",
    )
    parser.add_argument(
        "--group-errors",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_2025_12_group_errors.csv",
    )
    parser.add_argument("--test-month", default="2025-12")
    parser.add_argument("--min-group-rows", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--l2-leaf-reg", type=float, default=None)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    params = {"verbose": args.verbose, "thread_count": args.thread_count}
    if args.iterations is not None:
        params["iterations"] = args.iterations
    if args.depth is not None:
        params["depth"] = args.depth
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.l2_leaf_reg is not None:
        params["l2_leaf_reg"] = args.l2_leaf_reg

    report = BroadMonthlyFoldErrorAnalyzer().analyze_file(
        dataset_path=args.dataset,
        setup_path=args.setup,
        best_params_path=args.best_params,
        backtest_metrics_path=args.backtest_metrics,
        output_path=args.output,
        group_errors_path=args.group_errors,
        test_month=args.test_month,
        min_group_rows=args.min_group_rows,
        params=params,
    )
    catboost = report["overall"]["catboost"]
    baseline = report["overall"]["baseline"]
    print(
        "broad monthly fold error analysis complete: "
        f"test_month={report['test_month']}, "
        f"catboost_mape={catboost.get('mape'):.2f}%, "
        f"baseline_mape={baseline.get('mape'):.2f}%, "
        f"group_errors={args.group_errors}, "
        f"report={report['output_path']}"
    )


if __name__ == "__main__":
    main()
