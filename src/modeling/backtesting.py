"""Time-based backtesting for broad monthly CatBoost models."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.preprocessing import BROAD_MONTHLY_TARGET_COLUMN, GAP_TO_TARGET_MONTHS_COLUMN, PROCESSED_DATA_DIR

from .catboost_training import (
    DEFAULT_CATBOOST_PARAMS,
    BroadMonthlyCatBoostTrainer,
    _json_default,
    regression_metrics,
)


DEFAULT_BACKTEST_MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]


class BroadMonthlyCatBoostBacktester:
    """Run expanding-window backtesting on broad monthly salary data."""

    def backtest_file(
        self,
        dataset_path: str | Path = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet",
        setup_path: str | Path = PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json",
        best_params_path: str | Path | None = PROCESSED_DATA_DIR / "catboost_broad_monthly_best_params.json",
        metrics_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_metrics.csv",
        summary_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_summary.json",
        test_months: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run backtesting folds and save per-fold metrics plus summary."""
        from catboost import CatBoostRegressor, Pool

        dataset_path = Path(dataset_path)
        setup_path = Path(setup_path)
        metrics_path = Path(metrics_path)
        summary_path = Path(summary_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
        if not setup_path.exists():
            raise FileNotFoundError(f"Modeling setup not found: {setup_path}")

        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        print(f"[1/4] Loading dataset: {dataset_path}", flush=True)
        df = pd.read_parquet(dataset_path)
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        feature_columns = setup["feature_columns"]
        categorical_features = setup["catboost_cat_features"]
        target_column = setup.get("target_column", BROAD_MONTHLY_TARGET_COLUMN)
        period = pd.PeriodIndex(df[setup["period_column"]].astype("string"), freq="M")

        model_params = {
            **DEFAULT_CATBOOST_PARAMS,
            "verbose": False,
            "allow_writing_files": False,
        }
        if best_params_path is not None and Path(best_params_path).exists():
            best_report = json.loads(Path(best_params_path).read_text(encoding="utf-8"))
            model_params.update(best_report.get("best_params", {}))
        if params:
            model_params.update(params)

        test_months = test_months or DEFAULT_BACKTEST_MONTHS
        trainer = BroadMonthlyCatBoostTrainer()
        rows = []
        print(f"[2/4] Running folds: {', '.join(test_months)}", flush=True)
        for fold_index, test_month in enumerate(test_months, start=1):
            test_period = pd.Period(test_month, freq="M")
            validation_period = test_period - 1
            train_mask = period < validation_period
            validation_mask = period == validation_period
            test_mask = period == test_period
            train = df.loc[train_mask].copy()
            validation = df.loc[validation_mask].copy()
            test = df.loc[test_mask].copy()
            if train.empty or validation.empty or test.empty:
                print(
                    f"[3/4] Skipping fold {test_month}: "
                    f"train={len(train)}, validation={len(validation)}, test={len(test)}",
                    flush=True,
                )
                continue

            print(
                f"[3/4] Fold {fold_index}/{len(test_months)} test={test_month}: "
                f"train={len(train)}, validation={len(validation)}, test={len(test)}",
                flush=True,
            )
            fold_started = time.perf_counter()
            train_pool = trainer._make_pool(Pool, train, feature_columns, categorical_features, target_column)
            validation_pool = trainer._make_pool(Pool, validation, feature_columns, categorical_features, target_column)
            test_pool = trainer._make_pool(Pool, test, feature_columns, categorical_features, target_column)
            model = CatBoostRegressor(**model_params)
            model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
            prediction = model.predict(test_pool)

            catboost_metrics = regression_metrics(test[target_column], prediction)
            baseline_metrics = regression_metrics(test[target_column], test["median_salary_mid"])
            gap_one = test[GAP_TO_TARGET_MONTHS_COLUMN].eq(1)
            catboost_gap_one = regression_metrics(test.loc[gap_one, target_column], prediction[gap_one.to_numpy()])
            baseline_gap_one = regression_metrics(test.loc[gap_one, target_column], test.loc[gap_one, "median_salary_mid"])

            rows.append(
                {
                    "fold": fold_index,
                    "test_month": test_month,
                    "validation_month": str(validation_period),
                    "train_month_min": str(period[train_mask].min()),
                    "train_month_max": str(period[train_mask].max()),
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "test_rows": int(len(test)),
                    "gap_1_rows": int(gap_one.sum()),
                    "catboost_mae": catboost_metrics.get("mae"),
                    "catboost_rmse": catboost_metrics.get("rmse"),
                    "catboost_mape": catboost_metrics.get("mape"),
                    "catboost_smape": catboost_metrics.get("smape"),
                    "catboost_wape": catboost_metrics.get("wape"),
                    "catboost_ape_p90": catboost_metrics.get("ape_p90"),
                    "catboost_bias_mean_error": catboost_metrics.get("bias_mean_error"),
                    "baseline_mae": baseline_metrics.get("mae"),
                    "baseline_rmse": baseline_metrics.get("rmse"),
                    "baseline_mape": baseline_metrics.get("mape"),
                    "baseline_smape": baseline_metrics.get("smape"),
                    "baseline_wape": baseline_metrics.get("wape"),
                    "baseline_ape_p90": baseline_metrics.get("ape_p90"),
                    "baseline_bias_mean_error": baseline_metrics.get("bias_mean_error"),
                    "mape_delta_catboost_minus_baseline": (
                        catboost_metrics.get("mape") - baseline_metrics.get("mape")
                        if catboost_metrics.get("mape") is not None and baseline_metrics.get("mape") is not None
                        else None
                    ),
                    "gap_1_catboost_mape": catboost_gap_one.get("mape"),
                    "gap_1_baseline_mape": baseline_gap_one.get("mape"),
                    "gap_1_mape_delta_catboost_minus_baseline": (
                        catboost_gap_one.get("mape") - baseline_gap_one.get("mape")
                        if catboost_gap_one.get("mape") is not None and baseline_gap_one.get("mape") is not None
                        else None
                    ),
                    "best_iteration": model.get_best_iteration(),
                    "tree_count": model.tree_count_,
                    "duration_seconds": time.perf_counter() - fold_started,
                }
            )

        metrics = pd.DataFrame(rows)
        summary = self._build_summary(metrics)
        summary.update(
            {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.perf_counter() - started_perf,
                "dataset_path": str(dataset_path),
                "setup_path": str(setup_path),
                "best_params_path": str(best_params_path) if best_params_path is not None else None,
                "params": model_params,
                "metrics_path": str(metrics_path),
            }
        )

        print("[4/4] Saving backtest artifacts.", flush=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
        return summary

    def _build_summary(self, metrics: pd.DataFrame) -> dict[str, Any]:
        if metrics.empty:
            return {"fold_count": 0}
        catboost_better = metrics["mape_delta_catboost_minus_baseline"].lt(0)
        return {
            "fold_count": int(len(metrics)),
            "test_months": metrics["test_month"].tolist(),
            "mean_catboost_mape": float(metrics["catboost_mape"].mean()),
            "std_catboost_mape": float(metrics["catboost_mape"].std(ddof=0)),
            "worst_catboost_mape": float(metrics["catboost_mape"].max()),
            "worst_catboost_mape_month": str(metrics.loc[metrics["catboost_mape"].idxmax(), "test_month"]),
            "mean_baseline_mape": float(metrics["baseline_mape"].mean()),
            "mean_mape_delta_catboost_minus_baseline": float(metrics["mape_delta_catboost_minus_baseline"].mean()),
            "catboost_better_fold_count": int(catboost_better.sum()),
            "catboost_better_fold_share": float(catboost_better.mean()),
        }
