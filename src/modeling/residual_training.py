"""Residual CatBoost training for broad monthly salary forecasting."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import (
    BROAD_MONTHLY_TARGET_COLUMN,
    BROAD_QUARTERLY_TARGET_COLUMN,
    GAP_TO_TARGET_MONTHS_COLUMN,
    GAP_TO_TARGET_QUARTERS_COLUMN,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)

from .catboost_training import (
    DEFAULT_CATBOOST_PARAMS,
    BroadMonthlyCatBoostTrainer,
    _json_default,
    regression_metrics,
)


MODELS_DIR = PROJECT_ROOT / "models"
RESIDUAL_TARGET_COLUMN = "target_log_salary_delta"
RESIDUAL_PREDICTION_COLUMN = "prediction_log_salary_delta"
RESIDUAL_SALARY_PREDICTION_COLUMN = "prediction_residual_catboost"


@dataclass
class ResidualCatBoostTrainingResult:
    """Paths and metrics produced by residual CatBoost training."""

    metrics: dict[str, Any]
    model_path: Path
    metrics_path: Path
    params_path: Path
    feature_importance_path: Path
    predictions_path: Path
    residual_dataset_path: Path | None


class BroadMonthlyResidualCatBoostTrainer:
    """Train CatBoost to predict log target change over the salary baseline."""

    default_dataset_path = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet"
    default_setup_path = PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json"
    default_model_path = MODELS_DIR / "catboost_broad_monthly_residual_salary_model.cbm"
    default_params_path = PROCESSED_DATA_DIR / "catboost_broad_monthly_residual_params.json"
    default_metrics_path = PROCESSED_DATA_DIR / "catboost_broad_monthly_residual_metrics.json"
    default_feature_importance_path = PROCESSED_DATA_DIR / "feature_importance_broad_monthly_residual.csv"
    default_predictions_path = PROCESSED_DATA_DIR / "predictions_broad_monthly_residual_test.parquet"
    default_residual_dataset_path = PROCESSED_DATA_DIR / "ml_broad_monthly_residual_salary_dataset.parquet"
    default_salary_target_column = BROAD_MONTHLY_TARGET_COLUMN
    gap_column = GAP_TO_TARGET_MONTHS_COLUMN
    period_freq = "M"
    log_label = "monthly"

    def train_file(
        self,
        dataset_path: str | Path | None = None,
        setup_path: str | Path | None = None,
        model_path: str | Path | None = None,
        params_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
        feature_importance_path: str | Path | None = None,
        predictions_path: str | Path | None = None,
        residual_dataset_path: str | Path | None = None,
        params: dict[str, Any] | None = None,
    ) -> ResidualCatBoostTrainingResult:
        """Train residual CatBoost and save salary-space metrics/predictions."""
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        from catboost import CatBoostRegressor, Pool
        import catboost

        dataset_path = Path(dataset_path or self.default_dataset_path)
        setup_path = Path(setup_path or self.default_setup_path)
        model_path = Path(model_path or self.default_model_path)
        params_path = Path(params_path or self.default_params_path)
        metrics_path = Path(metrics_path or self.default_metrics_path)
        feature_importance_path = Path(feature_importance_path or self.default_feature_importance_path)
        predictions_path = Path(predictions_path or self.default_predictions_path)
        if residual_dataset_path is None:
            residual_dataset_path = self.default_residual_dataset_path
        print(f"[1/7] Loading dataset: {dataset_path}", flush=True)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
        if not setup_path.exists():
            raise FileNotFoundError(f"Modeling setup not found: {setup_path}")

        df = pd.read_parquet(dataset_path)
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        target_column = setup.get("target_column", self.default_salary_target_column)
        residual_df, residual_report = self._prepare_residual_dataframe(df, target_column)
        residual_setup = {**setup, "target_column": RESIDUAL_TARGET_COLUMN}
        train_params = {**DEFAULT_CATBOOST_PARAMS, **(params or {})}

        print(
            "[2/7] Residual target prepared: "
            f"rows={len(residual_df)}, dropped_rows={residual_report['dropped_row_count']}",
            flush=True,
        )
        trainer = BroadMonthlyCatBoostTrainer()
        split_frames = self._split_dataframe(residual_df, residual_setup)
        feature_columns = setup["feature_columns"]
        categorical_features = setup["catboost_cat_features"]
        print(
            "[3/7] Split rows: "
            f"train={len(split_frames['train'])}, "
            f"validation={len(split_frames['validation'])}, "
            f"test={len(split_frames['test'])}",
            flush=True,
        )

        train_pool = trainer._make_pool(
            Pool,
            split_frames["train"],
            feature_columns,
            categorical_features,
            RESIDUAL_TARGET_COLUMN,
        )
        validation_pool = trainer._make_pool(
            Pool,
            split_frames["validation"],
            feature_columns,
            categorical_features,
            RESIDUAL_TARGET_COLUMN,
        )
        test_pool = trainer._make_pool(
            Pool,
            split_frames["test"],
            feature_columns,
            categorical_features,
            RESIDUAL_TARGET_COLUMN,
        )

        print("[4/7] Training residual CatBoost.", flush=True)
        model = CatBoostRegressor(**train_params)
        model.fit(train_pool, eval_set=validation_pool, use_best_model=True)

        print("[5/7] Predicting train/validation/test splits.", flush=True)
        predictions = {
            split: model.predict(trainer._make_pool(Pool, frame, feature_columns, categorical_features))
            for split, frame in split_frames.items()
        }

        print("[6/7] Computing salary-space and residual-space metrics.", flush=True)
        metrics = self._build_metrics(
            split_frames=split_frames,
            predictions=predictions,
            target_column=target_column,
            params=train_params,
            setup=setup,
            residual_report=residual_report,
        )
        metrics["training_metadata"] = {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": float(time.perf_counter() - started_perf),
            "dataset_path": str(dataset_path),
            "setup_path": str(setup_path),
            "dataset_row_count": int(len(df)),
            "residual_dataset_row_count": int(len(residual_df)),
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

        residual_dataset_output = Path(residual_dataset_path) if residual_dataset_path is not None else None
        for path in [
            model_path,
            params_path,
            metrics_path,
            feature_importance_path,
            predictions_path,
            residual_dataset_output,
        ]:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)

        print("[7/7] Saving residual model artifacts.", flush=True)
        model.save_model(model_path)
        training_config = {
            "params": train_params,
            "dataset_path": str(dataset_path),
            "setup_path": str(setup_path),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "feature_importance_path": str(feature_importance_path),
            "predictions_path": str(predictions_path),
            "residual_dataset_path": str(residual_dataset_output) if residual_dataset_output is not None else None,
            "target_column": RESIDUAL_TARGET_COLUMN,
            "salary_target_column": target_column,
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

        test_predictions = self._prediction_frame(split_frames["test"], predictions["test"], target_column)
        test_predictions.to_parquet(predictions_path, index=False)
        if residual_dataset_output is not None:
            residual_df.to_parquet(residual_dataset_output, index=False)

        return ResidualCatBoostTrainingResult(
            metrics=metrics,
            model_path=model_path,
            metrics_path=metrics_path,
            params_path=params_path,
            feature_importance_path=feature_importance_path,
            predictions_path=predictions_path,
            residual_dataset_path=residual_dataset_output,
        )

    def _split_dataframe(self, df: pd.DataFrame, setup: dict[str, Any]) -> dict[str, pd.DataFrame]:
        period = pd.PeriodIndex(df[setup["period_column"]].astype("string"), freq=self.period_freq)
        train_min = pd.Period(setup["split"]["train"]["period_min"], freq=self.period_freq)
        train_max = pd.Period(setup["split"]["train"]["period_max"], freq=self.period_freq)
        validation_period = pd.Period(setup["split"]["validation"]["period"], freq=self.period_freq)
        test_period = pd.Period(setup["split"]["test"]["period"], freq=self.period_freq)

        return {
            "train": df.loc[(period >= train_min) & (period <= train_max)].copy(),
            "validation": df.loc[period == validation_period].copy(),
            "test": df.loc[period == test_period].copy(),
        }

    def _prepare_residual_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        baseline = pd.to_numeric(df["median_salary_mid"], errors="coerce")
        target = pd.to_numeric(df[target_column], errors="coerce")
        valid = baseline.gt(0) & target.gt(0)
        result = df.loc[valid].copy()
        result[RESIDUAL_TARGET_COLUMN] = np.log(result[target_column]) - np.log(result["median_salary_mid"])
        result["prediction_baseline"] = result["median_salary_mid"]
        return result, {
            "input_row_count": int(len(df)),
            "output_row_count": int(len(result)),
            "dropped_row_count": int((~valid).sum()),
            "target_column": target_column,
            "baseline_column": "median_salary_mid",
            "residual_target_column": RESIDUAL_TARGET_COLUMN,
            "residual_target_min": float(result[RESIDUAL_TARGET_COLUMN].min()) if len(result) else None,
            "residual_target_max": float(result[RESIDUAL_TARGET_COLUMN].max()) if len(result) else None,
            "residual_target_median": float(result[RESIDUAL_TARGET_COLUMN].median()) if len(result) else None,
        }

    def _build_metrics(
        self,
        split_frames: dict[str, pd.DataFrame],
        predictions: dict[str, Any],
        target_column: str,
        params: dict[str, Any],
        setup: dict[str, Any],
        residual_report: dict[str, Any],
    ) -> dict[str, Any]:
        by_split = {}
        for split, frame in split_frames.items():
            prediction_frame = self._prediction_frame(frame, predictions[split], target_column)
            by_split[split] = {
                "residual_catboost": regression_metrics(
                    prediction_frame[target_column],
                    prediction_frame[RESIDUAL_SALARY_PREDICTION_COLUMN],
                ),
                "baseline": regression_metrics(
                    prediction_frame[target_column],
                    prediction_frame["prediction_baseline"],
                ),
                "residual_target": self._residual_metrics(
                    prediction_frame[RESIDUAL_TARGET_COLUMN],
                    prediction_frame[RESIDUAL_PREDICTION_COLUMN],
                ),
            }

        test_predictions = self._prediction_frame(split_frames["test"], predictions["test"], target_column)
        gap_one = test_predictions[self.gap_column].eq(1)
        test_gap_one = {
            "residual_catboost": regression_metrics(
                test_predictions.loc[gap_one, target_column],
                test_predictions.loc[gap_one, RESIDUAL_SALARY_PREDICTION_COLUMN],
            ),
            "baseline": regression_metrics(
                test_predictions.loc[gap_one, target_column],
                test_predictions.loc[gap_one, "prediction_baseline"],
            ),
            "residual_target": self._residual_metrics(
                test_predictions.loc[gap_one, RESIDUAL_TARGET_COLUMN],
                test_predictions.loc[gap_one, RESIDUAL_PREDICTION_COLUMN],
            ),
        }

        return {
            "target_column": RESIDUAL_TARGET_COLUMN,
            "salary_target_column": target_column,
            "feature_columns": setup["feature_columns"],
            "categorical_features": setup["catboost_cat_features"],
            "params": params,
            "split": setup["split"],
            "residual_dataset": residual_report,
            "metrics_by_split": by_split,
            f"test_{self.gap_column}_1": test_gap_one,
        }

    def _prediction_frame(
        self,
        frame: pd.DataFrame,
        predicted_log_delta: Any,
        target_column: str,
    ) -> pd.DataFrame:
        result = frame.copy()
        result[RESIDUAL_PREDICTION_COLUMN] = predicted_log_delta
        result[RESIDUAL_SALARY_PREDICTION_COLUMN] = result["median_salary_mid"] * np.exp(
            result[RESIDUAL_PREDICTION_COLUMN]
        )
        result["prediction_baseline"] = result["median_salary_mid"]
        result["prediction_catboost"] = result[RESIDUAL_SALARY_PREDICTION_COLUMN]
        result["target_salary"] = result[target_column]
        return result

    def _residual_metrics(self, actual: Any, predicted: Any) -> dict[str, Any]:
        actual_series = pd.to_numeric(pd.Series(actual).reset_index(drop=True), errors="coerce")
        predicted_series = pd.to_numeric(pd.Series(predicted).reset_index(drop=True), errors="coerce")
        valid = actual_series.notna() & predicted_series.notna()
        actual_series = actual_series[valid]
        predicted_series = predicted_series[valid]
        if len(actual_series) == 0:
            return {"row_count": int(len(valid)), "valid_row_count": 0}
        error = predicted_series - actual_series
        absolute_error = error.abs()
        return {
            "row_count": int(len(valid)),
            "valid_row_count": int(len(actual_series)),
            "mae": float(absolute_error.mean()),
            "medae": float(absolute_error.median()),
            "rmse": float((error.pow(2).mean()) ** 0.5),
            "error_p90": float(absolute_error.quantile(0.90)),
            "bias_mean_error": float(error.mean()),
        }


class BroadQuarterlyResidualCatBoostTrainer(BroadMonthlyResidualCatBoostTrainer):
    """Train CatBoost to predict quarterly log salary change over the salary baseline."""

    default_dataset_path = PROCESSED_DATA_DIR / "ml_broad_quarterly_salary_dataset.parquet"
    default_setup_path = PROCESSED_DATA_DIR / "broad_quarterly_modeling_setup.json"
    default_model_path = MODELS_DIR / "catboost_broad_quarterly_residual_salary_model.cbm"
    default_params_path = PROCESSED_DATA_DIR / "catboost_broad_quarterly_residual_params.json"
    default_metrics_path = PROCESSED_DATA_DIR / "catboost_broad_quarterly_residual_metrics.json"
    default_feature_importance_path = PROCESSED_DATA_DIR / "feature_importance_broad_quarterly_residual.csv"
    default_predictions_path = PROCESSED_DATA_DIR / "predictions_broad_quarterly_residual_test.parquet"
    default_residual_dataset_path = PROCESSED_DATA_DIR / "ml_broad_quarterly_residual_salary_dataset.parquet"
    default_salary_target_column = BROAD_QUARTERLY_TARGET_COLUMN
    gap_column = GAP_TO_TARGET_QUARTERS_COLUMN
    period_freq = "Q"
    log_label = "quarterly"
