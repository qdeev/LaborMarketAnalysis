"""Hyperparameter tuning for broad monthly CatBoost models."""

from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.preprocessing import BROAD_MONTHLY_TARGET_COLUMN, PROCESSED_DATA_DIR

from .catboost_training import (
    DEFAULT_CATBOOST_PARAMS,
    BroadMonthlyCatBoostTrainer,
    _json_default,
    regression_metrics,
)


DEFAULT_TUNING_GRID: dict[str, list[Any]] = {
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05],
    "l2_leaf_reg": [3, 10, 25, 50],
    "iterations": [1000, 2000, 3000],
}


class BroadMonthlyCatBoostTuner:
    """Run validation-only CatBoost hyperparameter search."""

    def tune_file(
        self,
        dataset_path: str | Path = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet",
        setup_path: str | Path = PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json",
        results_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_tuning_results.csv",
        best_params_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_best_params.json",
        grid: dict[str, list[Any]] | None = None,
        max_runs: int | None = None,
        base_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run tuning against the setup validation split and save artifacts."""
        from catboost import CatBoostRegressor, Pool

        dataset_path = Path(dataset_path)
        setup_path = Path(setup_path)
        results_path = Path(results_path)
        best_params_path = Path(best_params_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
        if not setup_path.exists():
            raise FileNotFoundError(f"Modeling setup not found: {setup_path}")

        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        print(f"[1/5] Loading dataset: {dataset_path}", flush=True)
        df = pd.read_parquet(dataset_path)
        setup = json.loads(setup_path.read_text(encoding="utf-8"))

        trainer = BroadMonthlyCatBoostTrainer()
        split_frames = trainer._split_dataframe(df, setup)
        feature_columns = setup["feature_columns"]
        categorical_features = setup["catboost_cat_features"]
        target_column = setup.get("target_column", BROAD_MONTHLY_TARGET_COLUMN)
        print(
            "[2/5] Split rows: "
            f"train={len(split_frames['train'])}, "
            f"validation={len(split_frames['validation'])}",
            flush=True,
        )

        train_pool = trainer._make_pool(
            Pool,
            split_frames["train"],
            feature_columns,
            categorical_features,
            target_column,
        )
        validation_pool = trainer._make_pool(
            Pool,
            split_frames["validation"],
            feature_columns,
            categorical_features,
            target_column,
        )

        grid = grid or DEFAULT_TUNING_GRID
        candidates = list(self._iter_grid(grid))
        if max_runs is not None:
            candidates = candidates[:max_runs]
        print(f"[3/5] Running CatBoost tuning candidates: {len(candidates)}", flush=True)

        default_params = {
            **DEFAULT_CATBOOST_PARAMS,
            "verbose": False,
            "allow_writing_files": False,
        }
        if base_params:
            default_params.update(base_params)

        baseline_metrics = regression_metrics(
            split_frames["validation"][target_column],
            split_frames["validation"]["median_salary_mid"],
        )
        rows = []
        for index, candidate in enumerate(candidates, start=1):
            params = {**default_params, **candidate}
            run_started = time.perf_counter()
            print(f"[4/5] Candidate {index}/{len(candidates)}: {candidate}", flush=True)
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
            prediction = model.predict(validation_pool)
            metrics = regression_metrics(split_frames["validation"][target_column], prediction)
            rows.append(
                {
                    **candidate,
                    "validation_mae": metrics.get("mae"),
                    "validation_rmse": metrics.get("rmse"),
                    "validation_mape": metrics.get("mape"),
                    "validation_smape": metrics.get("smape"),
                    "validation_wape": metrics.get("wape"),
                    "validation_ape_p90": metrics.get("ape_p90"),
                    "validation_bias_mean_error": metrics.get("bias_mean_error"),
                    "best_iteration": model.get_best_iteration(),
                    "tree_count": model.tree_count_,
                    "duration_seconds": time.perf_counter() - run_started,
                }
            )

        results = pd.DataFrame(rows).sort_values("validation_mape", ascending=True)
        best_row = results.iloc[0].to_dict() if len(results) else {}
        best_params = {
            key: self._cast_like_grid_value(best_row[key], grid[key])
            for key in grid
            if key in best_row
        }
        finished_at = datetime.now(timezone.utc)
        best_report = {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": time.perf_counter() - started_perf,
            "dataset_path": str(dataset_path),
            "setup_path": str(setup_path),
            "results_path": str(results_path),
            "candidate_count": int(len(results)),
            "selection_metric": "validation_mape",
            "best_params": best_params,
            "best_validation_metrics": {
                key: self._clean_scalar(best_row.get(key))
                for key in [
                    "validation_mae",
                    "validation_rmse",
                    "validation_mape",
                    "validation_smape",
                    "validation_wape",
                    "validation_ape_p90",
                    "validation_bias_mean_error",
                ]
            },
            "baseline_validation_metrics": baseline_metrics,
        }

        print("[5/5] Saving tuning results.", flush=True)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_path, index=False)
        best_params_path.write_text(
            json.dumps(best_report, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return best_report

    def _iter_grid(self, grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        keys = list(grid)
        return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]

    def _clean_scalar(self, value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return value

    def _cast_like_grid_value(self, value: Any, grid_values: list[Any]) -> Any:
        clean_value = self._clean_scalar(value)
        if grid_values and isinstance(grid_values[0], int) and not isinstance(grid_values[0], bool):
            return int(clean_value)
        if grid_values and isinstance(grid_values[0], float):
            return float(clean_value)
        return clean_value
