"""Time-based backtesting for broad monthly residual CatBoost."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.preprocessing import (
    BROAD_MONTHLY_TARGET_COLUMN,
    BROAD_QUARTERLY_TARGET_COLUMN,
    GAP_TO_TARGET_MONTHS_COLUMN,
    GAP_TO_TARGET_QUARTERS_COLUMN,
    PROCESSED_DATA_DIR,
)

from .backtesting import DEFAULT_BACKTEST_MONTHS
from .catboost_training import (
    DEFAULT_CATBOOST_PARAMS,
    BroadMonthlyCatBoostTrainer,
    _json_default,
    regression_metrics,
)
from .ensemble import BroadMonthlyEnsembleEvaluator, BroadQuarterlyEnsembleEvaluator, DEFAULT_ALPHA_GRID
from .residual_training import (
    BroadMonthlyResidualCatBoostTrainer,
    BroadQuarterlyResidualCatBoostTrainer,
    RESIDUAL_TARGET_COLUMN,
)


DEFAULT_BACKTEST_QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]


class BroadMonthlyResidualCatBoostBacktester:
    """Run expanding-window backtesting for residual salary CatBoost."""

    default_dataset_path = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet"
    default_setup_path = PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json"
    default_best_params_path = PROCESSED_DATA_DIR / "catboost_broad_quarterly_backtest_best_params.json"
    default_metrics_path = PROCESSED_DATA_DIR / "catboost_broad_monthly_residual_backtest_metrics.csv"
    default_summary_path = PROCESSED_DATA_DIR / "catboost_broad_monthly_residual_backtest_summary.json"
    default_ensemble_metrics_path = PROCESSED_DATA_DIR / "catboost_broad_monthly_ensemble_backtest_metrics.csv"
    default_ensemble_summary_path = PROCESSED_DATA_DIR / "catboost_broad_monthly_ensemble_backtest_summary.json"
    default_test_periods = DEFAULT_BACKTEST_MONTHS
    default_salary_target_column = BROAD_MONTHLY_TARGET_COLUMN
    gap_column = GAP_TO_TARGET_MONTHS_COLUMN
    period_freq = "M"
    period_label = "month"
    residual_trainer_class = BroadMonthlyResidualCatBoostTrainer
    ensemble_evaluator_class = BroadMonthlyEnsembleEvaluator

    def backtest_file(
        self,
        dataset_path: str | Path | None = None,
        setup_path: str | Path | None = None,
        best_params_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
        summary_path: str | Path | None = None,
        ensemble_metrics_path: str | Path | None = None,
        ensemble_summary_path: str | Path | None = None,
        test_months: list[str] | None = None,
        test_periods: list[str] | None = None,
        params: dict[str, Any] | None = None,
        alpha_grid: list[float] | None = None,
    ) -> dict[str, Any]:
        """Run residual backtesting folds and save per-fold metrics plus summary."""
        from catboost import CatBoostRegressor, Pool

        dataset_path = Path(dataset_path or self.default_dataset_path)
        setup_path = Path(setup_path or self.default_setup_path)
        if best_params_path is None:
            best_params_path = self.default_best_params_path
        metrics_path = Path(metrics_path or self.default_metrics_path)
        summary_path = Path(summary_path or self.default_summary_path)
        if ensemble_metrics_path is None:
            ensemble_metrics_path = self.default_ensemble_metrics_path
        if ensemble_summary_path is None:
            ensemble_summary_path = self.default_ensemble_summary_path
        ensemble_metrics_path = Path(ensemble_metrics_path) if ensemble_metrics_path is not None else None
        ensemble_summary_path = Path(ensemble_summary_path) if ensemble_summary_path is not None else None
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
        if not setup_path.exists():
            raise FileNotFoundError(f"Modeling setup not found: {setup_path}")

        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        print(f"[1/4] Loading dataset: {dataset_path}", flush=True)
        df = pd.read_parquet(dataset_path)
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        target_column = setup.get("target_column", self.default_salary_target_column)
        residual_trainer = self.residual_trainer_class()
        df, residual_report = residual_trainer._prepare_residual_dataframe(df, target_column)
        feature_columns = setup["feature_columns"]
        categorical_features = setup["catboost_cat_features"]
        period = pd.PeriodIndex(df[setup["period_column"]].astype("string"), freq=self.period_freq)

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

        test_periods = test_periods or test_months or self.default_test_periods
        direct_trainer = BroadMonthlyCatBoostTrainer()
        ensemble_evaluator = self.ensemble_evaluator_class()
        alpha_grid = alpha_grid or DEFAULT_ALPHA_GRID
        rows = []
        ensemble_rows = []
        print(f"[2/4] Running residual folds: {', '.join(test_periods)}", flush=True)
        for fold_index, test_period_text in enumerate(test_periods, start=1):
            test_period = pd.Period(test_period_text, freq=self.period_freq)
            validation_period = test_period - 1
            train_mask = period < validation_period
            validation_mask = period == validation_period
            test_mask = period == test_period
            train = df.loc[train_mask].copy()
            validation = df.loc[validation_mask].copy()
            test = df.loc[test_mask].copy()
            if train.empty or validation.empty or test.empty:
                print(
                    f"[3/4] Skipping fold {test_period_text}: "
                    f"train={len(train)}, validation={len(validation)}, test={len(test)}",
                    flush=True,
                )
                continue

            print(
                f"[3/4] Fold {fold_index}/{len(test_periods)} test={test_period_text}: "
                f"train={len(train)}, validation={len(validation)}, test={len(test)}",
                flush=True,
            )
            fold_started = time.perf_counter()
            train_pool = direct_trainer._make_pool(Pool, train, feature_columns, categorical_features, RESIDUAL_TARGET_COLUMN)
            validation_pool = direct_trainer._make_pool(
                Pool,
                validation,
                feature_columns,
                categorical_features,
                RESIDUAL_TARGET_COLUMN,
            )
            test_pool = direct_trainer._make_pool(Pool, test, feature_columns, categorical_features, RESIDUAL_TARGET_COLUMN)
            model = CatBoostRegressor(**model_params)
            model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
            prediction_frame = residual_trainer._prediction_frame(test, model.predict(test_pool), target_column)
            _, fold_ensemble_metrics = ensemble_evaluator.evaluate_dataframe(prediction_frame, alpha_grid=alpha_grid)
            for ensemble_row in fold_ensemble_metrics.to_dict("records"):
                ensemble_rows.append(
                    {
                        "fold": fold_index,
                        f"test_{self.period_label}": test_period_text,
                        "test_month": test_period_text,
                        **ensemble_row,
                    }
                )

            residual_metrics = regression_metrics(
                prediction_frame[target_column],
                prediction_frame["prediction_residual_catboost"],
            )
            baseline_metrics = regression_metrics(prediction_frame[target_column], prediction_frame["prediction_baseline"])
            gap_one = prediction_frame[self.gap_column].eq(1)
            residual_gap_one = regression_metrics(
                prediction_frame.loc[gap_one, target_column],
                prediction_frame.loc[gap_one, "prediction_residual_catboost"],
            )
            baseline_gap_one = regression_metrics(
                prediction_frame.loc[gap_one, target_column],
                prediction_frame.loc[gap_one, "prediction_baseline"],
            )

            rows.append(
                {
                    "fold": fold_index,
                    f"test_{self.period_label}": test_period_text,
                    f"validation_{self.period_label}": str(validation_period),
                    f"train_{self.period_label}_min": str(period[train_mask].min()),
                    f"train_{self.period_label}_max": str(period[train_mask].max()),
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "test_rows": int(len(test)),
                    "gap_1_rows": int(gap_one.sum()),
                    "residual_catboost_mae": residual_metrics.get("mae"),
                    "residual_catboost_rmse": residual_metrics.get("rmse"),
                    "residual_catboost_mape": residual_metrics.get("mape"),
                    "residual_catboost_smape": residual_metrics.get("smape"),
                    "residual_catboost_wape": residual_metrics.get("wape"),
                    "residual_catboost_ape_p90": residual_metrics.get("ape_p90"),
                    "residual_catboost_bias_mean_error": residual_metrics.get("bias_mean_error"),
                    "baseline_mae": baseline_metrics.get("mae"),
                    "baseline_rmse": baseline_metrics.get("rmse"),
                    "baseline_mape": baseline_metrics.get("mape"),
                    "baseline_smape": baseline_metrics.get("smape"),
                    "baseline_wape": baseline_metrics.get("wape"),
                    "baseline_ape_p90": baseline_metrics.get("ape_p90"),
                    "baseline_bias_mean_error": baseline_metrics.get("bias_mean_error"),
                    "mape_delta_residual_catboost_minus_baseline": (
                        residual_metrics.get("mape") - baseline_metrics.get("mape")
                        if residual_metrics.get("mape") is not None and baseline_metrics.get("mape") is not None
                        else None
                    ),
                    "gap_1_residual_catboost_mape": residual_gap_one.get("mape"),
                    "gap_1_baseline_mape": baseline_gap_one.get("mape"),
                    "gap_1_mape_delta_residual_catboost_minus_baseline": (
                        residual_gap_one.get("mape") - baseline_gap_one.get("mape")
                        if residual_gap_one.get("mape") is not None and baseline_gap_one.get("mape") is not None
                        else None
                    ),
                    "best_iteration": model.get_best_iteration(),
                    "tree_count": model.tree_count_,
                    "duration_seconds": time.perf_counter() - fold_started,
                }
            )

        metrics = pd.DataFrame(rows)
        ensemble_metrics = pd.DataFrame(ensemble_rows)
        summary = self._build_summary(metrics)
        ensemble_summary = ensemble_evaluator.evaluate_backtest_metrics(ensemble_metrics)
        summary.update(
            {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.perf_counter() - started_perf,
                "dataset_path": str(dataset_path),
                "setup_path": str(setup_path),
                "best_params_path": str(best_params_path) if best_params_path is not None else None,
                "params": model_params,
                "period_label": self.period_label,
                "residual_dataset": residual_report,
                "metrics_path": str(metrics_path),
                "ensemble_metrics_path": str(ensemble_metrics_path) if ensemble_metrics_path is not None else None,
                "ensemble_summary_path": str(ensemble_summary_path) if ensemble_summary_path is not None else None,
            }
        )
        ensemble_summary.update(
            {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(dataset_path),
                "setup_path": str(setup_path),
                "alpha_grid": alpha_grid,
                "metrics_path": str(ensemble_metrics_path) if ensemble_metrics_path is not None else None,
            }
        )

        print("[4/4] Saving residual backtest artifacts.", flush=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        if ensemble_metrics_path is not None:
            ensemble_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            ensemble_metrics.to_csv(ensemble_metrics_path, index=False)
        if ensemble_summary_path is not None:
            ensemble_summary_path.parent.mkdir(parents=True, exist_ok=True)
            ensemble_summary_path.write_text(
                json.dumps(ensemble_summary, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
        summary["summary_path"] = str(summary_path)
        return summary

    def _build_summary(self, metrics: pd.DataFrame) -> dict[str, Any]:
        if metrics.empty:
            return {"fold_count": 0}
        residual_better = metrics["mape_delta_residual_catboost_minus_baseline"].lt(0)
        period_column = f"test_{self.period_label}"
        return {
            "fold_count": int(len(metrics)),
            f"test_{self.period_label}s": metrics[period_column].tolist(),
            "mean_residual_catboost_mape": float(metrics["residual_catboost_mape"].mean()),
            "std_residual_catboost_mape": float(metrics["residual_catboost_mape"].std(ddof=0)),
            "worst_residual_catboost_mape": float(metrics["residual_catboost_mape"].max()),
            f"worst_residual_catboost_mape_{self.period_label}": str(
                metrics.loc[metrics["residual_catboost_mape"].idxmax(), period_column]
            ),
            "mean_baseline_mape": float(metrics["baseline_mape"].mean()),
            "mean_mape_delta_residual_catboost_minus_baseline": float(
                metrics["mape_delta_residual_catboost_minus_baseline"].mean()
            ),
            "residual_catboost_better_fold_count": int(residual_better.sum()),
            "residual_catboost_better_fold_share": float(residual_better.mean()),
        }


class BroadQuarterlyResidualCatBoostBacktester(BroadMonthlyResidualCatBoostBacktester):
    """Run expanding-window backtesting for quarterly residual salary CatBoost."""

    default_dataset_path = PROCESSED_DATA_DIR / "ml_broad_quarterly_salary_dataset.parquet"
    default_setup_path = PROCESSED_DATA_DIR / "broad_quarterly_modeling_setup.json"
    default_best_params_path = PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_best_params.json"
    default_metrics_path = PROCESSED_DATA_DIR / "catboost_broad_quarterly_residual_backtest_metrics.csv"
    default_summary_path = PROCESSED_DATA_DIR / "catboost_broad_quarterly_residual_backtest_summary.json"
    default_ensemble_metrics_path = PROCESSED_DATA_DIR / "catboost_broad_quarterly_ensemble_backtest_metrics.csv"
    default_ensemble_summary_path = PROCESSED_DATA_DIR / "catboost_broad_quarterly_ensemble_backtest_summary.json"
    default_test_periods = DEFAULT_BACKTEST_QUARTERS
    default_salary_target_column = BROAD_QUARTERLY_TARGET_COLUMN
    gap_column = GAP_TO_TARGET_QUARTERS_COLUMN
    period_freq = "Q"
    period_label = "quarter"
    residual_trainer_class = BroadQuarterlyResidualCatBoostTrainer
    ensemble_evaluator_class = BroadQuarterlyEnsembleEvaluator
