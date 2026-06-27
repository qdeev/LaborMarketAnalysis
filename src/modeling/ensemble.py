"""Fallback and blending strategies for broad monthly salary predictions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd

from src.preprocessing import BROAD_MONTHLY_TARGET_COLUMN, BROAD_QUARTERLY_TARGET_COLUMN, PROCESSED_DATA_DIR

from .catboost_training import _json_default, regression_metrics


DEFAULT_ALPHA_GRID = [round(value / 10, 1) for value in range(0, 11)]


@dataclass
class EnsembleEvaluationResult:
    """Saved ensemble evaluation artifacts."""

    report: dict[str, Any]
    metrics: pd.DataFrame
    output_path: Path
    metrics_path: Path


class BroadMonthlyEnsembleEvaluator:
    """Evaluate baseline/CatBoost fallback and blending strategies."""

    target_column = BROAD_MONTHLY_TARGET_COLUMN
    catboost_prediction_column = "prediction_residual_catboost"
    baseline_prediction_column = "prediction_baseline"

    def evaluate_file(
        self,
        predictions_path: str | Path = PROCESSED_DATA_DIR / "predictions_broad_monthly_residual_test.parquet",
        output_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_ensemble_evaluation.json",
        metrics_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_ensemble_metrics.csv",
        alpha_grid: list[float] | None = None,
    ) -> EnsembleEvaluationResult:
        """Evaluate ensemble strategies on a predictions parquet."""
        predictions_path = Path(predictions_path)
        output_path = Path(output_path)
        metrics_path = Path(metrics_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions parquet not found: {predictions_path}")

        df = pd.read_parquet(predictions_path)
        report, metrics = self.evaluate_dataframe(df, alpha_grid=alpha_grid)
        report["predictions_path"] = str(predictions_path)
        report["metrics_path"] = str(metrics_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        metrics.to_csv(metrics_path, index=False)
        report["output_path"] = str(output_path)
        return EnsembleEvaluationResult(
            report=report,
            metrics=metrics,
            output_path=output_path,
            metrics_path=metrics_path,
        )

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        alpha_grid: list[float] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Evaluate all configured strategies on a prediction frame."""
        self._validate_columns(df)
        alpha_grid = alpha_grid or DEFAULT_ALPHA_GRID
        rows = []
        for strategy_name, prediction in self._strategy_predictions(df, alpha_grid).items():
            metrics = regression_metrics(df[self.target_column], prediction)
            baseline = regression_metrics(df[self.target_column], df[self.baseline_prediction_column])
            rows.append(
                {
                    "strategy": strategy_name,
                    "row_count": metrics.get("valid_row_count"),
                    "mae": metrics.get("mae"),
                    "medae": metrics.get("medae"),
                    "rmse": metrics.get("rmse"),
                    "mape": metrics.get("mape"),
                    "smape": metrics.get("smape"),
                    "wape": metrics.get("wape"),
                    "ape_p50": metrics.get("ape_p50"),
                    "ape_p75": metrics.get("ape_p75"),
                    "ape_p90": metrics.get("ape_p90"),
                    "bias_mean_error": metrics.get("bias_mean_error"),
                    "mape_delta_vs_baseline": (
                        metrics.get("mape") - baseline.get("mape")
                        if metrics.get("mape") is not None and baseline.get("mape") is not None
                        else None
                    ),
                }
            )
        metrics_frame = pd.DataFrame(rows).sort_values(["mape", "ape_p90"], ascending=True)
        best = metrics_frame.iloc[0].to_dict() if len(metrics_frame) else {}
        report = {
            "row_count": int(len(df)),
            "target_column": self.target_column,
            "catboost_prediction_column": self.catboost_prediction_column,
            "baseline_prediction_column": self.baseline_prediction_column,
            "alpha_grid": alpha_grid,
            "best_strategy": best,
            "strategies": metrics_frame.to_dict("records"),
        }
        return report, metrics_frame

    def evaluate_backtest_metrics(self, metrics: pd.DataFrame) -> dict[str, Any]:
        """Summarize per-fold ensemble metrics."""
        if metrics.empty:
            return {"strategy_count": 0}
        rows = []
        for strategy, group in metrics.groupby("strategy", dropna=False):
            better = group["mape_delta_vs_baseline"].lt(0)
            rows.append(
                {
                    "strategy": strategy,
                    "fold_count": int(len(group)),
                    "mean_mape": float(group["mape"].mean()),
                    "std_mape": float(group["mape"].std(ddof=0)),
                    "worst_mape": float(group["mape"].max()),
                    "worst_mape_month": str(group.loc[group["mape"].idxmax(), "test_month"]),
                    "mean_wape": float(group["wape"].mean()),
                    "mean_ape_p90": float(group["ape_p90"].mean()),
                    "mean_mape_delta_vs_baseline": float(group["mape_delta_vs_baseline"].mean()),
                    "better_fold_count": int(better.sum()),
                    "better_fold_share": float(better.mean()),
                }
            )
        summary = pd.DataFrame(rows).sort_values(["mean_mape", "worst_mape"], ascending=True)
        return {
            "strategy_count": int(len(summary)),
            "best_strategy": summary.iloc[0].to_dict() if len(summary) else {},
            "strategies": summary.to_dict("records"),
        }

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [
            column
            for column in [
                self.target_column,
                self.catboost_prediction_column,
                self.baseline_prediction_column,
            ]
            if column not in df.columns
        ]
        if missing:
            raise KeyError(f"Missing ensemble prediction columns: {missing}")

    def _strategy_predictions(self, df: pd.DataFrame, alpha_grid: list[float]) -> dict[str, Any]:
        catboost = df[self.catboost_prediction_column]
        baseline = df[self.baseline_prediction_column]
        predictions: dict[str, Any] = {
            "always_baseline": baseline,
            "always_catboost": catboost,
        }
        for alpha in alpha_grid:
            predictions[f"blend_alpha_{alpha:.1f}"] = alpha * catboost + (1 - alpha) * baseline

        if "segment_support_bucket" in df.columns:
            low_support = df["segment_support_bucket"].astype("string").isin(["very_low", "low"])
            predictions["choose_baseline_for_low_segment_support"] = catboost.where(~low_support, baseline)
            predictions["blend_0_5_for_low_segment_support"] = catboost.where(
                ~low_support,
                0.5 * catboost + 0.5 * baseline,
            )

        if "region_support_bucket" in df.columns:
            low_region_support = df["region_support_bucket"].astype("string").isin(["very_low", "low", "medium"])
            predictions["choose_baseline_for_low_region_support"] = catboost.where(~low_region_support, baseline)

        if "prediction_log_salary_delta" in df.columns:
            large_adjustment = df["prediction_log_salary_delta"].abs().gt(0.35)
            predictions["blend_0_5_for_large_log_delta"] = catboost.where(
                ~large_adjustment,
                0.5 * catboost + 0.5 * baseline,
            )
            predictions["choose_baseline_for_large_log_delta"] = catboost.where(~large_adjustment, baseline)

        return predictions


class BroadQuarterlyEnsembleEvaluator(BroadMonthlyEnsembleEvaluator):
    """Evaluate fallback and blending strategies for broad quarterly salary predictions."""

    target_column = BROAD_QUARTERLY_TARGET_COLUMN

    def _strategy_predictions(self, df: pd.DataFrame, alpha_grid: list[float]) -> dict[str, Any]:
        predictions = super()._strategy_predictions(df, alpha_grid)
        catboost = df[self.catboost_prediction_column]
        baseline = df[self.baseline_prediction_column]
        if "is_q4_to_q1" not in df.columns:
            return predictions

        q4_to_q1 = pd.to_numeric(df["is_q4_to_q1"], errors="coerce").eq(1)
        lower_prediction = pd.concat([catboost, baseline], axis=1).min(axis=1)
        predictions["q4_to_q1_min_catboost_baseline"] = catboost.where(~q4_to_q1, lower_prediction)
        predictions["q4_to_q1_cap_growth_0pct"] = catboost.where(~q4_to_q1, catboost.clip(upper=baseline))
        predictions["q4_to_q1_cap_growth_5pct"] = catboost.where(~q4_to_q1, catboost.clip(upper=baseline * 1.05))
        predictions["q4_to_q1_baseline_minus_5pct"] = catboost.where(~q4_to_q1, baseline * 0.95)
        predictions["q4_to_q1_baseline_minus_10pct"] = catboost.where(~q4_to_q1, baseline * 0.90)

        if "seasonal_transition_salary_ratio_mean" in df.columns:
            seasonal_ratio = pd.to_numeric(df["seasonal_transition_salary_ratio_mean"], errors="coerce")
            seasonal_prediction = baseline * seasonal_ratio
            predictions["q4_to_q1_historical_transition_mean"] = catboost.where(
                ~(q4_to_q1 & seasonal_prediction.notna()),
                seasonal_prediction,
            )
            predictions["q4_to_q1_blend_catboost_historical_mean_0_5"] = catboost.where(
                ~(q4_to_q1 & seasonal_prediction.notna()),
                0.5 * catboost + 0.5 * seasonal_prediction,
            )

        if "seasonal_transition_salary_ratio_median" in df.columns:
            seasonal_ratio = pd.to_numeric(df["seasonal_transition_salary_ratio_median"], errors="coerce")
            seasonal_prediction = baseline * seasonal_ratio
            predictions["q4_to_q1_historical_transition_median"] = catboost.where(
                ~(q4_to_q1 & seasonal_prediction.notna()),
                seasonal_prediction,
            )

        return predictions
