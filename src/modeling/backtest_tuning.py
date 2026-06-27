"""Backtesting-based hyperparameter tuning for broad monthly CatBoost."""

from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.preprocessing import BROAD_MONTHLY_TARGET_COLUMN, GAP_TO_TARGET_MONTHS_COLUMN, PROCESSED_DATA_DIR

from .backtesting import DEFAULT_BACKTEST_MONTHS
from .catboost_training import (
    DEFAULT_CATBOOST_PARAMS,
    BroadMonthlyCatBoostTrainer,
    _json_default,
    regression_metrics,
)


DEFAULT_BACKTEST_TUNING_GRID: dict[str, list[Any]] = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05],
    "l2_leaf_reg": [10, 25, 50, 100],
    "iterations": [1000, 2000],
}


class BroadMonthlyBacktestCatBoostTuner:
    """Tune CatBoost parameters by expanding-window backtesting folds."""

    def tune_file(
        self,
        dataset_path: str | Path = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet",
        setup_path: str | Path = PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json",
        seed_params_path: str | Path | None = PROCESSED_DATA_DIR / "catboost_broad_monthly_best_params.json",
        results_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_tuning_results.csv",
        best_params_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_best_params.json",
        test_months: list[str] | None = None,
        grid: dict[str, list[Any]] | None = None,
        max_runs: int | None = None,
        base_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run backtesting-based tuning and save candidate-level results."""
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
        feature_columns = setup["feature_columns"]
        categorical_features = setup["catboost_cat_features"]
        target_column = setup.get("target_column", BROAD_MONTHLY_TARGET_COLUMN)
        period = pd.PeriodIndex(df[setup["period_column"]].astype("string"), freq="M")
        test_months = test_months or DEFAULT_BACKTEST_MONTHS

        default_params = {
            **DEFAULT_CATBOOST_PARAMS,
            "verbose": False,
            "allow_writing_files": False,
        }
        if seed_params_path is not None and Path(seed_params_path).exists():
            seed_report = json.loads(Path(seed_params_path).read_text(encoding="utf-8"))
            default_params.update(seed_report.get("best_params", {}))
        if base_params:
            default_params.update(base_params)

        grid = grid or DEFAULT_BACKTEST_TUNING_GRID
        candidates = list(self._iter_grid(grid))
        if max_runs is not None:
            candidates = candidates[:max_runs]

        trainer = BroadMonthlyCatBoostTrainer()
        print(
            f"[2/5] Running backtesting tuning: candidates={len(candidates)}, "
            f"folds={', '.join(test_months)}",
            flush=True,
        )
        rows = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            params = {**default_params, **candidate}
            candidate_started = time.perf_counter()
            print(f"[3/5] Candidate {candidate_index}/{len(candidates)}: {candidate}", flush=True)
            fold_rows = []
            for fold_index, test_month in enumerate(test_months, start=1):
                fold_row = self._fit_fold(
                    CatBoostRegressor,
                    Pool,
                    trainer,
                    df,
                    period,
                    feature_columns,
                    categorical_features,
                    target_column,
                    test_month,
                    params,
                )
                if fold_row is None:
                    continue
                fold_row["fold"] = fold_index
                fold_rows.append(fold_row)

            fold_metrics = pd.DataFrame(fold_rows)
            summary = self._summarize_candidate(fold_metrics)
            rows.append(
                {
                    "candidate": candidate_index,
                    **candidate,
                    **summary,
                    "duration_seconds": time.perf_counter() - candidate_started,
                }
            )

        results = pd.DataFrame(rows)
        results = self._sort_results(results)
        best_row = results.iloc[0].to_dict() if len(results) else {}
        best_params = {
            key: self._cast_like_grid_value(best_row[key], grid[key])
            for key in grid
            if key in best_row
        }
        best_report = {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - started_perf,
            "dataset_path": str(dataset_path),
            "setup_path": str(setup_path),
            "seed_params_path": str(seed_params_path) if seed_params_path is not None else None,
            "results_path": str(results_path),
            "candidate_count": int(len(results)),
            "test_months": test_months,
            "selection_metric": "mean_catboost_mape_then_worst_fold_mape_then_better_fold_share",
            "best_params": best_params,
            "best_backtest_metrics": {
                key: self._clean_scalar(best_row.get(key))
                for key in [
                    "mean_catboost_mape",
                    "std_catboost_mape",
                    "worst_catboost_mape",
                    "mean_baseline_mape",
                    "mean_mape_delta_catboost_minus_baseline",
                    "catboost_better_fold_count",
                    "catboost_better_fold_share",
                    "mean_catboost_wape",
                    "mean_catboost_ape_p90",
                ]
            },
            "base_params": default_params,
        }

        print("[5/5] Saving backtesting tuning results.", flush=True)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_path, index=False)
        best_params_path.write_text(
            json.dumps(best_report, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return best_report

    def _fit_fold(
        self,
        model_class: Any,
        pool_class: Any,
        trainer: BroadMonthlyCatBoostTrainer,
        df: pd.DataFrame,
        period: pd.PeriodIndex,
        feature_columns: list[str],
        categorical_features: list[str],
        target_column: str,
        test_month: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        test_period = pd.Period(test_month, freq="M")
        validation_period = test_period - 1
        train_mask = period < validation_period
        validation_mask = period == validation_period
        test_mask = period == test_period
        train = df.loc[train_mask].copy()
        validation = df.loc[validation_mask].copy()
        test = df.loc[test_mask].copy()
        if train.empty or validation.empty or test.empty:
            return None

        train_pool = trainer._make_pool(pool_class, train, feature_columns, categorical_features, target_column)
        validation_pool = trainer._make_pool(pool_class, validation, feature_columns, categorical_features, target_column)
        test_pool = trainer._make_pool(pool_class, test, feature_columns, categorical_features, target_column)
        model = model_class(**params)
        model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
        prediction = model.predict(test_pool)

        catboost = regression_metrics(test[target_column], prediction)
        baseline = regression_metrics(test[target_column], test["median_salary_mid"])
        gap_one = test[GAP_TO_TARGET_MONTHS_COLUMN].eq(1)
        catboost_gap_one = regression_metrics(test.loc[gap_one, target_column], prediction[gap_one.to_numpy()])
        baseline_gap_one = regression_metrics(test.loc[gap_one, target_column], test.loc[gap_one, "median_salary_mid"])
        return {
            "test_month": test_month,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "catboost_mape": catboost.get("mape"),
            "catboost_wape": catboost.get("wape"),
            "catboost_ape_p90": catboost.get("ape_p90"),
            "catboost_bias_mean_error": catboost.get("bias_mean_error"),
            "baseline_mape": baseline.get("mape"),
            "baseline_wape": baseline.get("wape"),
            "baseline_ape_p90": baseline.get("ape_p90"),
            "baseline_bias_mean_error": baseline.get("bias_mean_error"),
            "mape_delta_catboost_minus_baseline": (
                catboost.get("mape") - baseline.get("mape")
                if catboost.get("mape") is not None and baseline.get("mape") is not None
                else None
            ),
            "gap_1_catboost_mape": catboost_gap_one.get("mape"),
            "gap_1_baseline_mape": baseline_gap_one.get("mape"),
            "best_iteration": model.get_best_iteration(),
            "tree_count": model.tree_count_,
        }

    def _summarize_candidate(self, metrics: pd.DataFrame) -> dict[str, Any]:
        if metrics.empty:
            return {"fold_count": 0}
        catboost_better = metrics["mape_delta_catboost_minus_baseline"].lt(0)
        return {
            "fold_count": int(len(metrics)),
            "mean_catboost_mape": float(metrics["catboost_mape"].mean()),
            "std_catboost_mape": float(metrics["catboost_mape"].std(ddof=0)),
            "worst_catboost_mape": float(metrics["catboost_mape"].max()),
            "worst_catboost_mape_month": str(metrics.loc[metrics["catboost_mape"].idxmax(), "test_month"]),
            "mean_baseline_mape": float(metrics["baseline_mape"].mean()),
            "mean_mape_delta_catboost_minus_baseline": float(metrics["mape_delta_catboost_minus_baseline"].mean()),
            "catboost_better_fold_count": int(catboost_better.sum()),
            "catboost_better_fold_share": float(catboost_better.mean()),
            "mean_catboost_wape": float(metrics["catboost_wape"].mean()),
            "mean_catboost_ape_p90": float(metrics["catboost_ape_p90"].mean()),
            "mean_gap_1_catboost_mape": float(metrics["gap_1_catboost_mape"].mean()),
            "mean_gap_1_baseline_mape": float(metrics["gap_1_baseline_mape"].mean()),
            "mean_best_iteration": float(metrics["best_iteration"].mean()),
            "mean_tree_count": float(metrics["tree_count"].mean()),
        }

    def _sort_results(self, results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return results
        return results.sort_values(
            ["mean_catboost_mape", "worst_catboost_mape", "catboost_better_fold_share"],
            ascending=[True, True, False],
        )

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
