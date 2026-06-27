"""Tune CatBoost hyperparameters for broad monthly salary forecasting."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import BroadMonthlyCatBoostTuner, DEFAULT_TUNING_GRID  # noqa: E402
from src.preprocessing import PROCESSED_DATA_DIR  # noqa: E402


def _parse_values(text: str, cast: type) -> list:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


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
        "--results",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_tuning_results.csv",
    )
    parser.add_argument(
        "--best-params",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_best_params.json",
    )
    parser.add_argument("--depths", default="4,6,8")
    parser.add_argument("--learning-rates", default="0.01,0.03,0.05")
    parser.add_argument("--l2-leaf-regs", default="3,10,25,50")
    parser.add_argument("--iterations", default="1000,2000,3000")
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

    base_params = {"verbose": args.verbose, "thread_count": args.thread_count}
    report = BroadMonthlyCatBoostTuner().tune_file(
        dataset_path=args.dataset,
        setup_path=args.setup,
        results_path=args.results,
        best_params_path=args.best_params,
        grid=grid or DEFAULT_TUNING_GRID,
        max_runs=args.max_runs,
        base_params=base_params,
    )
    best = report["best_validation_metrics"]
    print(
        "broad monthly CatBoost tuning complete: "
        f"candidates={report['candidate_count']}, "
        f"best_validation_mape={best.get('validation_mape'):.2f}%, "
        f"results={report['results_path']}, "
        f"best_params={args.best_params}"
    )


if __name__ == "__main__":
    main()
