"""Model evaluation helpers."""

from pathlib import Path
from typing import Any
import json

import pandas as pd

from .catboost_training import regression_metrics


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class BroadMonthlyModelEvaluator:
    """Evaluate broad monthly CatBoost predictions against baseline."""

    target_column = "target_median_salary_mid_next_observed_month"
    catboost_prediction_column = "prediction_catboost"
    baseline_prediction_column = "prediction_baseline"

    def evaluate_file(
        self,
        predictions_path: str | Path,
        output_path: str | Path,
        group_metrics_path: str | Path,
    ) -> dict[str, Any]:
        """Evaluate predictions parquet and save summary JSON and grouped CSV."""
        predictions_path = Path(predictions_path)
        output_path = Path(output_path)
        group_metrics_path = Path(group_metrics_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions parquet not found: {predictions_path}")

        df = pd.read_parquet(predictions_path)
        report, group_metrics = self.evaluate_dataframe(df)
        report["predictions_path"] = str(predictions_path)
        report["group_metrics_path"] = str(group_metrics_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        group_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        group_metrics.to_csv(group_metrics_path, index=False)
        report["output_path"] = str(output_path)
        return report

    def evaluate_dataframe(self, df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        """Evaluate overall and grouped prediction quality."""
        required = [
            self.target_column,
            self.catboost_prediction_column,
            self.baseline_prediction_column,
            "source_dataset",
            "occupation_group",
            "region",
            "gap_to_target_months",
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Missing prediction columns: {missing}")

        report = {
            "row_count": int(len(df)),
            "overall": self._compare(df),
            "gap_to_target_months_1": self._compare(df[df["gap_to_target_months"].eq(1)]),
            "gap_to_target_months_gt_1": self._compare(df[df["gap_to_target_months"].gt(1)]),
            "by_source_dataset": self._group_report(df, "source_dataset"),
            "by_occupation_group": self._group_report(df, "occupation_group"),
            "by_region_top_worst_catboost_mape": self._group_report(
                df,
                "region",
                min_rows=10,
                sort_by="catboost_mape",
                limit=20,
            ),
            "catboost_better_row_count": int(
                (
                    (df[self.catboost_prediction_column] - df[self.target_column]).abs()
                    < (df[self.baseline_prediction_column] - df[self.target_column]).abs()
                ).sum()
            ),
        }
        group_metrics = pd.concat(
            [
                self._group_metrics_frame(df, "source_dataset"),
                self._group_metrics_frame(df, "occupation_group"),
                self._group_metrics_frame(df, "region"),
                self._group_metrics_frame(df, "gap_to_target_months"),
            ],
            ignore_index=True,
        )
        return report, group_metrics

    def _compare(self, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "catboost": regression_metrics(
                frame[self.target_column],
                frame[self.catboost_prediction_column],
            ),
            "baseline": regression_metrics(
                frame[self.target_column],
                frame[self.baseline_prediction_column],
            ),
        }

    def _group_report(
        self,
        df: pd.DataFrame,
        group_column: str,
        min_rows: int = 1,
        sort_by: str = "row_count",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        grouped = self._group_metrics_frame(df, group_column)
        grouped = grouped[grouped["row_count"] >= min_rows]
        grouped = grouped.sort_values(sort_by, ascending=False)
        if limit is not None:
            grouped = grouped.head(limit)
        return grouped.to_dict("records")

    def _group_metrics_frame(self, df: pd.DataFrame, group_column: str) -> pd.DataFrame:
        rows = []
        for value, group in df.groupby(group_column, dropna=False):
            catboost = regression_metrics(group[self.target_column], group[self.catboost_prediction_column])
            baseline = regression_metrics(group[self.target_column], group[self.baseline_prediction_column])
            rows.append(
                {
                    "group_column": group_column,
                    "group_value": str(value),
                    "row_count": int(len(group)),
                    "catboost_mae": catboost.get("mae"),
                    "catboost_medae": catboost.get("medae"),
                    "catboost_rmse": catboost.get("rmse"),
                    "catboost_mape": catboost.get("mape"),
                    "catboost_smape": catboost.get("smape"),
                    "catboost_wape": catboost.get("wape"),
                    "catboost_ape_p50": catboost.get("ape_p50"),
                    "catboost_ape_p75": catboost.get("ape_p75"),
                    "catboost_ape_p90": catboost.get("ape_p90"),
                    "catboost_bias_mean_error": catboost.get("bias_mean_error"),
                    "baseline_mae": baseline.get("mae"),
                    "baseline_medae": baseline.get("medae"),
                    "baseline_rmse": baseline.get("rmse"),
                    "baseline_mape": baseline.get("mape"),
                    "baseline_smape": baseline.get("smape"),
                    "baseline_wape": baseline.get("wape"),
                    "baseline_ape_p50": baseline.get("ape_p50"),
                    "baseline_ape_p75": baseline.get("ape_p75"),
                    "baseline_ape_p90": baseline.get("ape_p90"),
                    "baseline_bias_mean_error": baseline.get("bias_mean_error"),
                    "mape_delta_catboost_minus_baseline": (
                        catboost.get("mape") - baseline.get("mape")
                        if catboost.get("mape") is not None and baseline.get("mape") is not None
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows)
