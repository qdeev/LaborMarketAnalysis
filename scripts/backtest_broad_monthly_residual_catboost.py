"""Run time-based backtesting for broad monthly residual CatBoost."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import BroadMonthlyResidualCatBoostBacktester, DEFAULT_BACKTEST_MONTHS  # noqa: E402
from src.preprocessing import PROCESSED_DATA_DIR  # noqa: E402


def _parse_months(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def _parse_alpha_grid(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


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
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_best_params.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_residual_backtest_metrics.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_residual_backtest_summary.json",
    )
    parser.add_argument(
        "--ensemble-metrics",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_ensemble_backtest_metrics.csv",
    )
    parser.add_argument(
        "--ensemble-summary",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_ensemble_backtest_summary.json",
    )
    parser.add_argument("--test-months", default=",".join(DEFAULT_BACKTEST_MONTHS))
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
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

    summary = BroadMonthlyResidualCatBoostBacktester().backtest_file(
        dataset_path=args.dataset,
        setup_path=args.setup,
        best_params_path=args.best_params,
        metrics_path=args.metrics,
        summary_path=args.summary,
        ensemble_metrics_path=args.ensemble_metrics,
        ensemble_summary_path=args.ensemble_summary,
        test_months=_parse_months(args.test_months),
        params=params,
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
    )
    print(
        "broad monthly residual backtest complete: "
        f"folds={summary['fold_count']}, "
        f"mean_residual_catboost_mape={summary.get('mean_residual_catboost_mape'):.2f}%, "
        f"mean_baseline_mape={summary.get('mean_baseline_mape'):.2f}%, "
        f"better_folds={summary.get('residual_catboost_better_fold_count')}/{summary['fold_count']}, "
        f"metrics={args.metrics}, "
        f"summary={summary['summary_path']}"
    )


if __name__ == "__main__":
    main()
