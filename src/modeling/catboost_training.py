"""CatBoost training for broad monthly salary forecasting."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import pandas as pd

from src.preprocessing import (
    BROAD_MONTHLY_TARGET_COLUMN,
    GAP_TO_TARGET_MONTHS_COLUMN,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)


MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "eval_metric": "MAPE",
    "random_seed": 42,
    "early_stopping_rounds": 100,
    "verbose": 100,
    "allow_writing_files": False,
}


@dataclass
class CatBoostTrainingResult:
    """Paths and metrics produced by CatBoost training."""

    metrics: dict[str, Any]
    model_path: Path
    metrics_path: Path
    params_path: Path
    feature_importance_path: Path
    predictions_path: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def regression_metrics(actual: Any, predicted: Any) -> dict[str, Any]:
    """Compute core regression metrics in rubles and percent."""
    actual_series = pd.to_numeric(pd.Series(actual).reset_index(drop=True), errors="coerce")
    predicted_series = pd.to_numeric(pd.Series(predicted).reset_index(drop=True), errors="coerce")
    valid = actual_series.notna() & predicted_series.notna()
    actual_series = actual_series[valid]
    predicted_series = predicted_series[valid]
    if len(actual_series) == 0:
        return {"row_count": int(len(valid)), "valid_row_count": 0}

    error = predicted_series - actual_series
    absolute_error = error.abs()
    denominator = actual_series.abs()
    mape = absolute_error.div(denominator).replace([float("inf")], pd.NA).dropna()
    smape_denominator = actual_series.abs() + predicted_series.abs()
    smape = (2 * absolute_error).div(smape_denominator).replace([float("inf")], pd.NA).dropna()
    actual_sum = actual_series.abs().sum()

    return {
        "row_count": int(len(valid)),
        "valid_row_count": int(len(actual_series)),
        "mae": float(absolute_error.mean()),
        "medae": float(absolute_error.median()),
        "rmse": float((error.pow(2).mean()) ** 0.5),
        "mape": float(mape.mean() * 100) if len(mape) else None,
        "smape": float(smape.mean() * 100) if len(smape) else None,
        "wape": float(absolute_error.sum() / actual_sum * 100) if actual_sum else None,
        "ape_p50": float(mape.quantile(0.50) * 100) if len(mape) else None,
        "ape_p75": float(mape.quantile(0.75) * 100) if len(mape) else None,
        "ape_p90": float(mape.quantile(0.90) * 100) if len(mape) else None,
        "bias_mean_error": float(error.mean()),
    }


class BroadMonthlyCatBoostTrainer:
    """Train and evaluate CatBoost on the broad monthly salary dataset."""

    def train_file(
        self,
        dataset_path: str | Path = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet",
        setup_path: str | Path = PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json",
        model_path: str | Path = MODELS_DIR / "catboost_broad_monthly_salary_model.cbm",
        params_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_params.json",
        metrics_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_metrics.json",
        feature_importance_path: str | Path = PROCESSED_DATA_DIR / "feature_importance_broad_monthly.csv",
        predictions_path: str | Path = PROCESSED_DATA_DIR / "predictions_broad_monthly_test.parquet",
        params: dict[str, Any] | None = None,
    ) -> CatBoostTrainingResult:
        """Train CatBoost using a prepared dataset/setup and save artifacts."""
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        from catboost import CatBoostRegressor, Pool
        import catboost

        dataset_path = Path(dataset_path)
        setup_path = Path(setup_path)
        print(f"[1/6] Loading dataset: {dataset_path}", flush=True)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
        if not setup_path.exists():
            raise FileNotFoundError(f"Modeling setup not found: {setup_path}")

        df = pd.read_parquet(dataset_path)
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        train_params = {**DEFAULT_CATBOOST_PARAMS, **(params or {})}

        print(f"[2/6] Preparing split and CatBoost pools: rows={len(df)}", flush=True)
        split_frames = self._split_dataframe(df, setup)
        feature_columns = setup["feature_columns"]
        categorical_features = setup["catboost_cat_features"]
        target_column = setup.get("target_column", BROAD_MONTHLY_TARGET_COLUMN)
        print(
            "[2/6] Split rows: "
            f"train={len(split_frames['train'])}, "
            f"validation={len(split_frames['validation'])}, "
            f"test={len(split_frames['test'])}",
            flush=True,
        )

        train_pool = self._make_pool(
            Pool,
            split_frames["train"],
            feature_columns,
            categorical_features,
            target_column,
        )
        validation_pool = self._make_pool(
            Pool,
            split_frames["validation"],
            feature_columns,
            categorical_features,
            target_column,
        )
        test_pool = self._make_pool(
            Pool,
            split_frames["test"],
            feature_columns,
            categorical_features,
            target_column,
        )

        model = CatBoostRegressor(**train_params)
        print("[3/6] Training CatBoost. Iteration progress is printed by CatBoost below.", flush=True)
        model.fit(train_pool, eval_set=validation_pool, use_best_model=True)

        print("[4/6] Predicting train/validation/test splits.", flush=True)
        predictions = {
            split: model.predict(self._make_pool(Pool, frame, feature_columns, categorical_features))
            for split, frame in split_frames.items()
        }

        print("[5/6] Computing metrics.", flush=True)
        metrics = self._build_metrics(
            split_frames=split_frames,
            predictions=predictions,
            target_column=target_column,
            params=train_params,
            setup=setup,
        )
        finished_at = datetime.now(timezone.utc)
        training_duration_seconds = time.perf_counter() - started_perf
        metrics["training_metadata"] = {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": float(training_duration_seconds),
            "dataset_path": str(dataset_path),
            "setup_path": str(setup_path),
            "dataset_row_count": int(len(df)),
            "split_row_counts": {split: int(len(frame)) for split, frame in split_frames.items()},
            "feature_count": int(len(feature_columns)),
            "categorical_feature_count": int(len(categorical_features)),
            "best_iteration": int(model.get_best_iteration()) if model.get_best_iteration() is not None else None,
            "best_score": model.get_best_score(),
            "tree_count": int(model.tree_count_),
            "catboost_version": catboost.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
        }

        model_path = Path(model_path)
        params_path = Path(params_path)
        metrics_path = Path(metrics_path)
        feature_importance_path = Path(feature_importance_path)
        predictions_path = Path(predictions_path)
        for path in [model_path, params_path, metrics_path, feature_importance_path, predictions_path]:
            path.parent.mkdir(parents=True, exist_ok=True)

        print("[6/6] Saving model, metrics, params, feature importance, and test predictions.", flush=True)
        model.save_model(model_path)
        training_config = {
            "params": train_params,
            "dataset_path": str(dataset_path),
            "setup_path": str(setup_path),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "feature_importance_path": str(feature_importance_path),
            "predictions_path": str(predictions_path),
            "started_at_utc": started_at.isoformat(),
        }
        params_path.write_text(
            json.dumps(training_config, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.get_feature_importance(train_pool),
            }
        ).sort_values("importance", ascending=False)
        importance.to_csv(feature_importance_path, index=False)

        test_predictions = split_frames["test"].copy()
        test_predictions["prediction_catboost"] = predictions["test"]
        test_predictions["prediction_baseline"] = test_predictions["median_salary_mid"]
        test_predictions.to_parquet(predictions_path, index=False)

        return CatBoostTrainingResult(
            metrics=metrics,
            model_path=model_path,
            metrics_path=metrics_path,
            params_path=params_path,
            feature_importance_path=feature_importance_path,
            predictions_path=predictions_path,
        )

    def _split_dataframe(self, df: pd.DataFrame, setup: dict[str, Any]) -> dict[str, pd.DataFrame]:
        period = pd.PeriodIndex(df[setup["period_column"]].astype("string"), freq="M")
        train_min = pd.Period(setup["split"]["train"]["period_min"], freq="M")
        train_max = pd.Period(setup["split"]["train"]["period_max"], freq="M")
        validation_period = pd.Period(setup["split"]["validation"]["period"], freq="M")
        test_period = pd.Period(setup["split"]["test"]["period"], freq="M")

        return {
            "train": df.loc[(period >= train_min) & (period <= train_max)].copy(),
            "validation": df.loc[period == validation_period].copy(),
            "test": df.loc[period == test_period].copy(),
        }

    def _make_pool(
        self,
        pool_class: Any,
        frame: pd.DataFrame,
        feature_columns: list[str],
        categorical_features: list[str],
        target_column: str | None = None,
    ) -> Any:
        features = frame[feature_columns].copy()
        for column in categorical_features:
            features[column] = features[column].astype("string").fillna("__MISSING__")
        label = None if target_column is None else frame[target_column]
        return pool_class(features, label=label, cat_features=categorical_features)

    def _build_metrics(
        self,
        split_frames: dict[str, pd.DataFrame],
        predictions: dict[str, Any],
        target_column: str,
        params: dict[str, Any],
        setup: dict[str, Any],
    ) -> dict[str, Any]:
        by_split = {}
        for split, frame in split_frames.items():
            by_split[split] = {
                "catboost": regression_metrics(frame[target_column], predictions[split]),
                "baseline": regression_metrics(frame[target_column], frame["median_salary_mid"]),
            }

        test = split_frames["test"]
        gap_one = test[GAP_TO_TARGET_MONTHS_COLUMN].eq(1)
        test_gap_one = {
            "catboost": regression_metrics(test.loc[gap_one, target_column], predictions["test"][gap_one.to_numpy()]),
            "baseline": regression_metrics(
                test.loc[gap_one, target_column],
                test.loc[gap_one, "median_salary_mid"],
            ),
        }

        return {
            "target_column": target_column,
            "feature_columns": setup["feature_columns"],
            "categorical_features": setup["catboost_cat_features"],
            "params": params,
            "split": setup["split"],
            "metrics_by_split": by_split,
            "test_gap_to_target_months_1": test_gap_one,
        }
