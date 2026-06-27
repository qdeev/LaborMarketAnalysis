"""Tune broad monthly CatBoost parameters by time-based backtesting."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (  # noqa: E402
    BroadMonthlyBacktestCatBoostTuner,
    DEFAULT_BACKTEST_MONTHS,
    DEFAULT_BACKTEST_TUNING_GRID,
)
from src.preprocessing import PROCESSED_DATA_DIR  # noqa: E402


def _parse_values(text: str, cast: type) -> list:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def _parse_months(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


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
        "--seed-params",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_best_params.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_tuning_results.csv",
    )
    parser.add_argument(
        "--best-params",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_best_params.json",
    )
    parser.add_argument("--test-months", default=",".join(DEFAULT_BACKTEST_MONTHS))
    parser.add_argument("--depths", default=",".join(str(value) for value in DEFAULT_BACKTEST_TUNING_GRID["depth"]))
    parser.add_argument(
        "--learning-rates",
        default=",".join(str(value) for value in DEFAULT_BACKTEST_TUNING_GRID["learning_rate"]),
    )
    parser.add_argument(
        "--l2-leaf-regs",
        default=",".join(str(value) for value in DEFAULT_BACKTEST_TUNING_GRID["l2_leaf_reg"]),
    )
    parser.add_argument(
        "--iterations",
        default=",".join(str(value) for value in DEFAULT_BACKTEST_TUNING_GRID["iterations"]),
    )
    parser.add_argument("--bagging-temperatures", default=None)
    parser.add_argument("--random-strengths", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    grid = {
        "depth": _parse_values(args.depths, int),
        "learning_rate": _parse_values(args.learning_rates, float),
        "l2_leaf_reg": _parse_values(args.l2_leaf_regs, float),
        "iterations": _parse_values(args.iterations, int),
    }
    if args.bagging_temperatures:
        grid["bagging_temperature"] = _parse_values(args.bagging_temperatures, float)
    if args.random_strengths:
        grid["random_strength"] = _parse_values(args.random_strengths, float)

    base_params = {
        "verbose": args.verbose,
        "thread_count": args.thread_count,
    }
    report = BroadMonthlyBacktestCatBoostTuner().tune_file(
        dataset_path=args.dataset,
        setup_path=args.setup,
        seed_params_path=args.seed_params,
        results_path=args.results,
        best_params_path=args.best_params,
        test_months=_parse_months(args.test_months),
        grid=grid,
        max_runs=args.max_runs,
        base_params=base_params,
    )
    best = report["best_backtest_metrics"]
    print(
        "broad monthly CatBoost backtesting tuning complete: "
        f"candidates={report['candidate_count']}, "
        f"best_mean_mape={best.get('mean_catboost_mape'):.2f}%, "
        f"best_worst_fold_mape={best.get('worst_catboost_mape'):.2f}%, "
        f"better_folds_share={best.get('catboost_better_fold_share'):.2f}, "
        f"results={report['results_path']}, "
        f"best_params={args.best_params}"
    )


if __name__ == "__main__":
    main()
