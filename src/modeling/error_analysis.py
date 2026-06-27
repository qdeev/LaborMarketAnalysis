"""Error analysis helpers for broad monthly CatBoost folds."""

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


class BroadMonthlyFoldErrorAnalyzer:
    """Train one backtesting fold and explain where its errors come from."""

    def analyze_file(
        self,
        dataset_path: str | Path = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet",
        setup_path: str | Path = PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json",
        best_params_path: str | Path | None = PROCESSED_DATA_DIR / "catboost_broad_monthly_best_params.json",
        backtest_metrics_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_backtest_metrics.csv",
        output_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_2025_12_error_analysis.json",
        group_errors_path: str | Path = PROCESSED_DATA_DIR / "catboost_broad_monthly_2025_12_group_errors.csv",
        test_month: str = "2025-12",
        min_group_rows: int = 5,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run fold-level error diagnostics and save JSON/CSV artifacts."""
        from catboost import CatBoostRegressor, Pool

        dataset_path = Path(dataset_path)
        setup_path = Path(setup_path)
        backtest_metrics_path = Path(backtest_metrics_path)
        output_path = Path(output_path)
        group_errors_path = Path(group_errors_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
        if not setup_path.exists():
            raise FileNotFoundError(f"Modeling setup not found: {setup_path}")

        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        print(f"[1/5] Loading dataset: {dataset_path}", flush=True)
        df = pd.read_parquet(dataset_path)
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        target_column = setup.get("target_column", BROAD_MONTHLY_TARGET_COLUMN)
        period_column = setup["period_column"]
        period = pd.PeriodIndex(df[period_column].astype("string"), freq="M")
        test_period = pd.Period(test_month, freq="M")
        validation_period = test_period - 1

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

        print(f"[2/5] Preparing fold test={test_month}, validation={validation_period}", flush=True)
        train = df.loc[period < validation_period].copy()
        validation = df.loc[period == validation_period].copy()
        test = df.loc[period == test_period].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError(
                f"Fold has empty split: train={len(train)}, validation={len(validation)}, test={len(test)}"
            )

        trainer = BroadMonthlyCatBoostTrainer()
        feature_columns = setup["feature_columns"]
        categorical_features = setup["catboost_cat_features"]
        train_pool = trainer._make_pool(Pool, train, feature_columns, categorical_features, target_column)
        validation_pool = trainer._make_pool(Pool, validation, feature_columns, categorical_features, target_column)
        test_pool = trainer._make_pool(Pool, test, feature_columns, categorical_features, target_column)

        print("[3/5] Training one diagnostic fold.", flush=True)
        model = CatBoostRegressor(**model_params)
        model.fit(train_pool, eval_set=validation_pool, use_best_model=True)

        print("[4/5] Computing row and group errors.", flush=True)
        predictions = test.copy()
        predictions["prediction_catboost"] = model.predict(test_pool)
        predictions["prediction_baseline"] = predictions["median_salary_mid"]
        predictions = self._add_error_columns(predictions, target_column)

        group_errors = pd.concat(
            [
                self._group_errors(predictions, ["source_dataset"], target_column, min_group_rows),
                self._group_errors(predictions, ["region"], target_column, min_group_rows),
                self._group_errors(predictions, ["occupation_group"], target_column, min_group_rows),
                self._group_errors(
                    predictions,
                    ["source_dataset", "region", "occupation_group"],
                    target_column,
                    min_group_rows,
                ),
                self._group_errors(predictions, [GAP_TO_TARGET_MONTHS_COLUMN], target_column, 1),
            ],
            ignore_index=True,
        ).sort_values(["catboost_abs_error_share", "catboost_mape"], ascending=False)

        report = {
            "test_month": test_month,
            "validation_month": str(validation_period),
            "split_rows": {
                "train": int(len(train)),
                "validation": int(len(validation)),
                "test": int(len(test)),
            },
            "overall": {
                "catboost": regression_metrics(predictions[target_column], predictions["prediction_catboost"]),
                "baseline": regression_metrics(predictions[target_column], predictions["prediction_baseline"]),
            },
            "gap_to_target_months_1": self._compare(predictions[predictions[GAP_TO_TARGET_MONTHS_COLUMN].eq(1)], target_column),
            "fold_metrics_from_backtest": self._load_backtest_fold(backtest_metrics_path, test_month),
            "period_profiles": self._period_profiles(df, period, target_column, test_period),
            "largest_error_groups": self._top_records(group_errors, "catboost_abs_error_share", 20),
            "worst_mape_groups": self._top_records(group_errors[group_errors["row_count"] >= min_group_rows], "catboost_mape", 20),
            "groups_where_catboost_loses_most": self._top_records(
                group_errors[group_errors["row_count"] >= min_group_rows],
                "mape_delta_catboost_minus_baseline",
                20,
            ),
            "largest_row_errors": self._largest_row_errors(predictions, target_column),
            "support_slices": self._support_slices(predictions, target_column),
            "diagnostic_notes": self._diagnostic_notes(predictions, group_errors),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - started_perf,
            "dataset_path": str(dataset_path),
            "setup_path": str(setup_path),
            "best_params_path": str(best_params_path) if best_params_path is not None else None,
            "params": model_params,
            "group_errors_path": str(group_errors_path),
            "best_iteration": model.get_best_iteration(),
            "tree_count": model.tree_count_,
        }

        print("[5/5] Saving diagnostics.", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        group_errors_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        group_errors.to_csv(group_errors_path, index=False)
        report["output_path"] = str(output_path)
        return report

    def _add_error_columns(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        result = df.copy()
        result["catboost_error"] = result["prediction_catboost"] - result[target_column]
        result["baseline_error"] = result["prediction_baseline"] - result[target_column]
        result["catboost_abs_error"] = result["catboost_error"].abs()
        result["baseline_abs_error"] = result["baseline_error"].abs()
        result["catboost_ape"] = result["catboost_abs_error"] / result[target_column].abs() * 100
        result["baseline_ape"] = result["baseline_abs_error"] / result[target_column].abs() * 100
        result["catboost_better_than_baseline"] = result["catboost_abs_error"] < result["baseline_abs_error"]
        result["salary_change_pct"] = (result[target_column] - result["median_salary_mid"]) / result["median_salary_mid"].abs() * 100
        return result

    def _group_errors(
        self,
        df: pd.DataFrame,
        group_columns: list[str],
        target_column: str,
        min_rows: int,
    ) -> pd.DataFrame:
        rows = []
        total_abs_error = df["catboost_abs_error"].sum()
        for keys, group in df.groupby(group_columns, dropna=False):
            if len(group) < min_rows:
                continue
            if not isinstance(keys, tuple):
                keys = (keys,)
            catboost = regression_metrics(group[target_column], group["prediction_catboost"])
            baseline = regression_metrics(group[target_column], group["prediction_baseline"])
            row = {
                "group_columns": " + ".join(group_columns),
                "group_value": " | ".join(str(value) for value in keys),
                "row_count": int(len(group)),
                "catboost_mae": catboost.get("mae"),
                "catboost_mape": catboost.get("mape"),
                "catboost_wape": catboost.get("wape"),
                "catboost_ape_p90": catboost.get("ape_p90"),
                "catboost_bias_mean_error": catboost.get("bias_mean_error"),
                "baseline_mae": baseline.get("mae"),
                "baseline_mape": baseline.get("mape"),
                "baseline_wape": baseline.get("wape"),
                "baseline_ape_p90": baseline.get("ape_p90"),
                "baseline_bias_mean_error": baseline.get("bias_mean_error"),
                "mape_delta_catboost_minus_baseline": (
                    catboost.get("mape") - baseline.get("mape")
                    if catboost.get("mape") is not None and baseline.get("mape") is not None
                    else None
                ),
                "catboost_abs_error_sum": float(group["catboost_abs_error"].sum()),
                "catboost_abs_error_share": (
                    float(group["catboost_abs_error"].sum() / total_abs_error) if total_abs_error else None
                ),
                "catboost_better_row_share": float(group["catboost_better_than_baseline"].mean()),
                "median_salary_mid_median": float(group["median_salary_mid"].median()),
                "target_median_salary_mid_median": float(group[target_column].median()),
                "salary_change_pct_median": float(group["salary_change_pct"].median()),
                "vacancy_count_median": float(group["vacancy_count"].median()),
                "salary_count_median": float(group["salary_count"].median()),
                "data_quality_score_median": float(group["data_quality_score"].median()),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def _compare(self, frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
        return {
            "catboost": regression_metrics(frame[target_column], frame["prediction_catboost"]),
            "baseline": regression_metrics(frame[target_column], frame["prediction_baseline"]),
        }

    def _load_backtest_fold(self, path: Path, test_month: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        metrics = pd.read_csv(path)
        fold = metrics[metrics["test_month"].astype("string").eq(test_month)]
        if fold.empty:
            return None
        return fold.iloc[0].to_dict()

    def _period_profiles(
        self,
        df: pd.DataFrame,
        period: pd.PeriodIndex,
        target_column: str,
        test_period: pd.Period,
    ) -> dict[str, Any]:
        months = [test_period - 1, test_period, test_period + 1]
        profiles = {}
        for month in months:
            frame = df.loc[period == month]
            profiles[str(month)] = self._profile_period(frame, target_column)
        return profiles

    def _profile_period(self, frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
        if frame.empty:
            return {"row_count": 0}
        return {
            "row_count": int(len(frame)),
            "median_salary_mid_median": float(frame["median_salary_mid"].median()),
            "target_median_salary_mid_median": float(frame[target_column].median()),
            "vacancy_count_median": float(frame["vacancy_count"].median()),
            "salary_count_median": float(frame["salary_count"].median()),
            "data_quality_score_median": float(frame["data_quality_score"].median()),
            "source_dataset": self._value_distribution(frame, "source_dataset", 10),
            "region": self._value_distribution(frame, "region", 15),
            "occupation_group": self._value_distribution(frame, "occupation_group", 15),
            GAP_TO_TARGET_MONTHS_COLUMN: self._value_distribution(frame, GAP_TO_TARGET_MONTHS_COLUMN, 15),
        }

    def _value_distribution(self, df: pd.DataFrame, column: str, limit: int) -> list[dict[str, Any]]:
        values = df[column].value_counts(dropna=False).head(limit)
        total = len(df)
        return [
            {"value": str(index), "row_count": int(count), "share": float(count / total)}
            for index, count in values.items()
        ]

    def _top_records(self, df: pd.DataFrame, sort_by: str, limit: int) -> list[dict[str, Any]]:
        if df.empty or sort_by not in df.columns:
            return []
        return df.sort_values(sort_by, ascending=False).head(limit).to_dict("records")

    def _largest_row_errors(self, df: pd.DataFrame, target_column: str, limit: int = 25) -> list[dict[str, Any]]:
        columns = [
            "source_dataset",
            "region",
            "occupation_group",
            "vacancy_month",
            "target_vacancy_month",
            GAP_TO_TARGET_MONTHS_COLUMN,
            "median_salary_mid",
            target_column,
            "prediction_catboost",
            "prediction_baseline",
            "catboost_abs_error",
            "baseline_abs_error",
            "catboost_ape",
            "baseline_ape",
            "salary_change_pct",
            "vacancy_count",
            "salary_count",
            "data_quality_score",
        ]
        return df.sort_values("catboost_abs_error", ascending=False).head(limit)[columns].to_dict("records")

    def _support_slices(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        slices = {
            "salary_count_eq_1": df["salary_count"].eq(1),
            "salary_count_lte_2": df["salary_count"].le(2),
            "vacancy_count_lte_2": df["vacancy_count"].le(2),
            "data_quality_score_lt_0_5": df["data_quality_score"].lt(0.5),
            "abs_salary_change_pct_gt_50": df["salary_change_pct"].abs().gt(50),
            "catboost_ape_gt_50": df["catboost_ape"].gt(50),
        }
        return {
            name: {
                "row_count": int(mask.sum()),
                "share": float(mask.mean()),
                "metrics": self._compare(df.loc[mask], target_column),
            }
            for name, mask in slices.items()
        }

    def _diagnostic_notes(self, df: pd.DataFrame, group_errors: pd.DataFrame) -> list[str]:
        notes = []
        if not group_errors.empty:
            worst_share = group_errors.sort_values("catboost_abs_error_share", ascending=False).iloc[0]
            notes.append(
                "Largest error contribution: "
                f"{worst_share['group_columns']}={worst_share['group_value']} "
                f"with {worst_share['catboost_abs_error_share']:.1%} of CatBoost absolute error."
            )
        better_share = df["catboost_better_than_baseline"].mean()
        notes.append(f"CatBoost beats baseline on {better_share:.1%} of rows in this fold.")
        high_ape_share = df["catboost_ape"].gt(50).mean()
        notes.append(f"Rows with CatBoost APE > 50%: {high_ape_share:.1%}.")
        return notes
