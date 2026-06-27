"""Train CatBoost for broad monthly salary forecasting."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import BroadMonthlyCatBoostTrainer, DEFAULT_CATBOOST_PARAMS  # noqa: E402
from src.preprocessing import PROCESSED_DATA_DIR, PROJECT_ROOT as SRC_PROJECT_ROOT  # noqa: E402


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
        "--model",
        type=Path,
        default=SRC_PROJECT_ROOT / "models" / "catboost_broad_monthly_salary_model.cbm",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_params.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROCESSED_DATA_DIR / "catboost_broad_monthly_metrics.json",
    )
    parser.add_argument(
        "--feature-importance",
        type=Path,
        default=PROCESSED_DATA_DIR / "feature_importance_broad_monthly.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROCESSED_DATA_DIR / "predictions_broad_monthly_test.parquet",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_CATBOOST_PARAMS["iterations"])
    parser.add_argument("--depth", type=int, default=DEFAULT_CATBOOST_PARAMS["depth"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CATBOOST_PARAMS["learning_rate"])
    parser.add_argument("--l2-leaf-reg", type=float, default=DEFAULT_CATBOOST_PARAMS.get("l2_leaf_reg"))
    parser.add_argument("--bagging-temperature", type=float, default=None)
    parser.add_argument("--random-strength", type=float, default=None)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_CATBOOST_PARAMS["early_stopping_rounds"],
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_CATBOOST_PARAMS["random_seed"])
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=DEFAULT_CATBOOST_PARAMS["verbose"])
    args = parser.parse_args()

    params = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "early_stopping_rounds": args.early_stopping_rounds,
        "random_seed": args.random_seed,
        "thread_count": args.thread_count,
        "verbose": args.verbose,
    }
    if args.l2_leaf_reg is not None:
        params["l2_leaf_reg"] = args.l2_leaf_reg
    if args.bagging_temperature is not None:
        params["bagging_temperature"] = args.bagging_temperature
    if args.random_strength is not None:
        params["random_strength"] = args.random_strength
    result = BroadMonthlyCatBoostTrainer().train_file(
        dataset_path=args.dataset,
        setup_path=args.setup,
        model_path=args.model,
        params_path=args.params,
        metrics_path=args.metrics,
        feature_importance_path=args.feature_importance,
        predictions_path=args.predictions,
        params=params,
    )
    test_metrics = result.metrics["metrics_by_split"]["test"]
    catboost_mape = test_metrics["catboost"].get("mape")
    baseline_mape = test_metrics["baseline"].get("mape")
    print(
        "broad monthly CatBoost trained: "
        f"catboost_test_mape={catboost_mape:.2f}%, "
        f"baseline_test_mape={baseline_mape:.2f}%, "
        f"model={result.model_path}, "
        f"metrics={result.metrics_path}"
    )


if __name__ == "__main__":
    main()
