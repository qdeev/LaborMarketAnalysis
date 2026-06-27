"""ML dataset readiness helpers for monthly salary forecasting."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .constants import PROCESSED_DATA_DIR


ML_SEGMENT_KEY_COLUMNS = [
    "source_dataset",
    "country",
    "region",
    "occupation_group",
    "occupation_code",
    "experience_group",
    "employment_type",
    "schedule_type",
]

BROAD_QUARTERLY_SEGMENT_KEY_COLUMNS = [
    "source_dataset",
    "region",
    "occupation_group",
]

BROAD_MONTHLY_SEGMENT_KEY_COLUMNS = BROAD_QUARTERLY_SEGMENT_KEY_COLUMNS


@dataclass
class MLReadinessResult:
    """Monthly ML readiness report and optional saved path."""

    report: dict[str, Any]
    output_path: Path | None = None


@dataclass
class MonthlyMLDatasetResult:
    """Prepared supervised ML dataset and optional saved artifact paths."""

    dataframe: Any
    report: dict[str, Any]
    output_path: Path | None = None
    report_path: Path | None = None


@dataclass
class QuarterlyMLDatasetResult:
    """Prepared supervised quarterly ML dataset and optional saved artifact paths."""

    dataframe: Any
    report: dict[str, Any]
    output_path: Path | None = None
    report_path: Path | None = None


TARGET_COLUMN = "target_median_salary_mid_t_plus_1"
TARGET_MONTH_COLUMN = "target_vacancy_month"
QUARTERLY_TARGET_COLUMN = "target_median_salary_mid_t_plus_1_quarter"
TARGET_QUARTER_COLUMN = "target_vacancy_quarter"
BROAD_MONTHLY_TARGET_COLUMN = "target_median_salary_mid_next_observed_month"
GAP_TO_TARGET_MONTHS_COLUMN = "gap_to_target_months"
BROAD_QUARTERLY_TARGET_COLUMN = "target_median_salary_mid_next_observed_quarter"
GAP_TO_TARGET_QUARTERS_COLUMN = "gap_to_target_quarters"
BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN = "target_log_salary_delta"

LAG_FEATURES = {
    "median_salary_mid_lag_1": ("median_salary_mid", 1),
    "median_salary_mid_lag_2": ("median_salary_mid", 2),
    "median_salary_from_lag_1": ("median_salary_from", 1),
    "median_salary_to_lag_1": ("median_salary_to", 1),
    "vacancy_count_lag_1": ("vacancy_count", 1),
    "salary_count_lag_1": ("salary_count", 1),
    "salary_missing_rate_lag_1": ("salary_missing_rate", 1),
    "employer_count_lag_1": ("employer_count", 1),
    "data_quality_score_lag_1": ("data_quality_score", 1),
}

QUARTERLY_LAG_FEATURES = {
    "median_salary_mid_lag_1q": ("median_salary_mid", 1),
    "median_salary_mid_lag_2q": ("median_salary_mid", 2),
    "vacancy_count_lag_1q": ("vacancy_count", 1),
    "salary_count_lag_1q": ("salary_count", 1),
    "salary_missing_rate_lag_1q": ("salary_missing_rate", 1),
    "employer_count_lag_1q": ("employer_count", 1),
    "data_quality_score_lag_1q": ("data_quality_score", 1),
    "months_observed_in_quarter_lag_1q": ("months_observed_in_quarter", 1),
}

BROAD_QUARTERLY_LAG_FEATURES = {
    "median_salary_mid_lag_1_observed": ("median_salary_mid", 1),
    "median_salary_mid_lag_2_observed": ("median_salary_mid", 2),
    "vacancy_count_lag_1_observed": ("vacancy_count", 1),
    "salary_count_lag_1_observed": ("salary_count", 1),
    "data_quality_score_lag_1_observed": ("data_quality_score", 1),
}

BROAD_QUARTERLY_DYNAMIC_FEATURES = [
    "median_salary_mid_delta_lag_1",
    "median_salary_mid_pct_change_lag_1",
    "median_salary_mid_lag_1_to_lag_2_delta",
    "median_salary_mid_lag_1_to_lag_2_pct_change",
    "segment_previous_observation_count",
    "segment_previous_salary_mean",
    "segment_previous_salary_median",
    "segment_previous_salary_std",
    "vacancy_count_delta_lag_1",
    "vacancy_count_pct_change_lag_1",
    "salary_count_delta_lag_1",
    "salary_count_pct_change_lag_1",
    "employer_count_delta_lag_1",
    "employer_count_pct_change_lag_1",
    "data_quality_score_delta_lag_1",
    "data_quality_score_pct_change_lag_1",
]

BROAD_QUARTERLY_CATEGORICAL_FEATURES = [
    "source_dataset",
    "region",
    "occupation_group",
    "quarter_transition",
]

BROAD_QUARTERLY_NUMERIC_FEATURES = [
    "vacancy_count",
    "salary_count",
    "employer_count",
    "months_observed_in_quarter",
    "median_salary_mid",
    "mean_salary_mid",
    "p25_salary_mid",
    "p75_salary_mid",
    "median_salary_from",
    "median_salary_to",
    "salary_missing_rate",
    "remote_share",
    "full_time_share",
    "shift_share",
    "top_employer_share",
    "data_quality_score",
    "gap_from_previous_observed_quarter",
    "median_salary_mid_lag_1_observed",
    "median_salary_mid_lag_2_observed",
    "vacancy_count_lag_1_observed",
    "salary_count_lag_1_observed",
    "data_quality_score_lag_1_observed",
    *BROAD_QUARTERLY_DYNAMIC_FEATURES,
    "quarter",
    "target_quarter",
    "year",
    "target_year",
    "quarter_index",
    "target_quarter_index",
    "is_q4_to_q1",
    "seasonal_transition_previous_count",
    "seasonal_transition_salary_ratio_mean",
    "seasonal_transition_salary_ratio_median",
    "seasonal_transition_log_delta_mean",
]

BROAD_MONTHLY_LAG_FEATURES = {
    "median_salary_mid_lag_1_observed": ("median_salary_mid", 1),
    "median_salary_mid_lag_2_observed": ("median_salary_mid", 2),
    "vacancy_count_lag_1_observed": ("vacancy_count", 1),
    "salary_count_lag_1_observed": ("salary_count", 1),
    "data_quality_score_lag_1_observed": ("data_quality_score", 1),
}

BROAD_MONTHLY_DYNAMIC_FEATURES = [
    "median_salary_mid_delta_lag_1",
    "median_salary_mid_pct_change_lag_1",
    "median_salary_mid_lag_1_to_lag_2_delta",
    "median_salary_mid_lag_1_to_lag_2_pct_change",
    "segment_previous_observation_count",
    "segment_previous_salary_mean",
    "segment_previous_salary_median",
    "segment_previous_salary_std",
    "vacancy_count_delta_lag_1",
    "vacancy_count_pct_change_lag_1",
    "salary_count_delta_lag_1",
    "salary_count_pct_change_lag_1",
    "employer_count_delta_lag_1",
    "employer_count_pct_change_lag_1",
    "data_quality_score_delta_lag_1",
    "data_quality_score_pct_change_lag_1",
]

BROAD_MONTHLY_REGION_RARE_THRESHOLD = 60
BROAD_MONTHLY_OCCUPATION_RARE_THRESHOLD = 100

BROAD_MONTHLY_SUPPORT_FEATURES = [
    "region_observation_count",
    "occupation_group_observation_count",
    "segment_observation_count",
]

BROAD_MONTHLY_TARGET_EXCLUDE_COLUMNS = [
    BROAD_MONTHLY_TARGET_COLUMN,
    TARGET_MONTH_COLUMN,
    GAP_TO_TARGET_MONTHS_COLUMN,
]

BROAD_MONTHLY_CATEGORICAL_FEATURES = [
    "source_dataset",
    "region",
    "occupation_group",
]

BROAD_MONTHLY_NUMERIC_FEATURES = [
    "vacancy_count",
    "salary_count",
    "employer_count",
    "median_salary_mid",
    "mean_salary_mid",
    "p25_salary_mid",
    "p75_salary_mid",
    "median_salary_from",
    "median_salary_to",
    "salary_missing_rate",
    "remote_share",
    "full_time_share",
    "shift_share",
    "top_employer_share",
    "data_quality_score",
    "gap_from_previous_observed_month",
    "median_salary_mid_lag_1_observed",
    "median_salary_mid_lag_2_observed",
    "vacancy_count_lag_1_observed",
    "salary_count_lag_1_observed",
    "data_quality_score_lag_1_observed",
    *BROAD_MONTHLY_DYNAMIC_FEATURES,
    "month",
    "quarter",
    "year",
    "month_index",
]


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("MonthlyMLReadinessAuditor requires pandas.") from exc
    return pd


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _value_counts(series: Any) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _weighted_average(group: Any, value_column: str, weight_column: str) -> float:
    pd = _require_pandas()
    values = pd.to_numeric(group[value_column], errors="coerce")
    weights = pd.to_numeric(group[weight_column], errors="coerce").fillna(0)
    valid = values.notna() & weights.gt(0)
    if valid.any():
        return float((values[valid] * weights[valid]).sum() / weights[valid].sum())
    return float(values.mean()) if values.notna().any() else float("nan")


def _first_non_null(series: Any) -> Any:
    values = series.dropna()
    return values.iloc[0] if len(values) else None


class MonthlyMLReadinessAuditor:
    """Audit canonical monthly segments before supervised ML preparation."""

    def audit_file(
        self,
        input_path: str | Path = PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
        output_path: str | Path | None = None,
    ) -> MLReadinessResult:
        """Load monthly segments parquet, audit it, and optionally save JSON."""
        pd = _require_pandas()
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Monthly segments parquet not found: {path}")

        df = pd.read_parquet(path)
        report = self.audit_dataframe(df)
        report["input_path"] = str(path)

        saved_path = Path(output_path) if output_path is not None else None
        if saved_path is not None:
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
        return MLReadinessResult(report=report, output_path=saved_path)

    def audit_dataframe(self, df: Any) -> dict[str, Any]:
        """Build readiness statistics for monthly salary forecasting."""
        pd = _require_pandas()
        missing_required = [
            column
            for column in [*ML_SEGMENT_KEY_COLUMNS, "vacancy_month", "median_salary_mid"]
            if column not in df.columns
        ]
        if missing_required:
            return {
                "is_ready_for_next_step": False,
                "missing_required_columns": missing_required,
            }

        work = df.copy()
        work["_vacancy_month_period"] = pd.PeriodIndex(work["vacancy_month"].astype("string"), freq="M")
        work = work.sort_values(ML_SEGMENT_KEY_COLUMNS + ["_vacancy_month_period"])
        next_month = work.groupby(ML_SEGMENT_KEY_COLUMNS, dropna=False)["_vacancy_month_period"].shift(-1)
        next_target = work.groupby(ML_SEGMENT_KEY_COLUMNS, dropna=False)["median_salary_mid"].shift(-1)
        expected_next_month = work["_vacancy_month_period"] + 1
        has_next_month_row = next_month.eq(expected_next_month)
        has_next_month_target = has_next_month_row & next_target.notna()

        source_next_target = (
            work.assign(_has_next_month_target=has_next_month_target)
            .groupby("source_dataset", dropna=False)["_has_next_month_target"]
            .sum()
            .astype(int)
            .to_dict()
        )

        month_counts = work.groupby("_vacancy_month_period", dropna=False).size()
        salary_missing_rate = float(work["median_salary_mid"].isna().mean()) if len(work) else 0.0

        return {
            "is_ready_for_next_step": bool(has_next_month_target.any()),
            "row_count": int(len(work)),
            "segment_key_columns": ML_SEGMENT_KEY_COLUMNS,
            "source_dataset_distribution": _value_counts(work["source_dataset"]),
            "month_min": str(work["_vacancy_month_period"].min()) if len(work) else None,
            "month_max": str(work["_vacancy_month_period"].max()) if len(work) else None,
            "month_count": int(work["_vacancy_month_period"].nunique()) if len(work) else 0,
            "rows_by_month": {str(key): int(value) for key, value in month_counts.items()},
            "median_salary_mid_missing_count": int(work["median_salary_mid"].isna().sum()),
            "median_salary_mid_missing_rate": salary_missing_rate,
            "rows_with_next_month_row": int(has_next_month_row.sum()),
            "rows_with_next_month_target": int(has_next_month_target.sum()),
            "rows_with_next_month_target_by_source": {
                str(key): int(value) for key, value in source_next_target.items()
            },
            "unique_segments": int(
                work[ML_SEGMENT_KEY_COLUMNS].drop_duplicates().shape[0]
            ),
        }


class MonthlySalaryMLDatasetBuilder:
    """Prepare a supervised t+1 monthly salary forecasting dataset."""

    def prepare_file(
        self,
        input_path: str | Path = PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
        output_path: str | Path | None = PROCESSED_DATA_DIR / "ml_monthly_salary_dataset.parquet",
        report_path: str | Path | None = PROCESSED_DATA_DIR / "ml_monthly_salary_dataset_report.json",
    ) -> MonthlyMLDatasetResult:
        """Load monthly segments, prepare supervised rows, and optionally save outputs."""
        pd = _require_pandas()
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Monthly segments parquet not found: {path}")

        df = pd.read_parquet(path)
        result = self.prepare_dataframe(df)
        result.report["input_path"] = str(path)

        saved_output = Path(output_path) if output_path is not None else None
        saved_report = Path(report_path) if report_path is not None else None

        if saved_output is not None:
            saved_output.parent.mkdir(parents=True, exist_ok=True)
            result.dataframe.to_parquet(saved_output, index=False)
        if saved_report is not None:
            saved_report.parent.mkdir(parents=True, exist_ok=True)
            saved_report.write_text(
                json.dumps(result.report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

        result.output_path = saved_output
        result.report_path = saved_report
        return result

    def prepare_dataframe(self, df: Any) -> MonthlyMLDatasetResult:
        """Prepare supervised rows from canonical monthly segments."""
        pd = _require_pandas()
        missing_required = [
            column
            for column in [
                *ML_SEGMENT_KEY_COLUMNS,
                "vacancy_month",
                "median_salary_mid",
                "salary_count",
                "vacancy_count",
                "data_quality_score",
            ]
            if column not in df.columns
        ]
        if missing_required:
            raise KeyError(f"Missing required monthly ML columns: {missing_required}")

        work = df.copy()
        work["_vacancy_month_period"] = pd.PeriodIndex(
            work["vacancy_month"].astype("string"),
            freq="M",
        )
        work = work.sort_values(ML_SEGMENT_KEY_COLUMNS + ["_vacancy_month_period"]).reset_index(drop=True)
        grouped = work.groupby(ML_SEGMENT_KEY_COLUMNS, dropna=False)

        next_month = grouped["_vacancy_month_period"].shift(-1)
        next_target = grouped["median_salary_mid"].shift(-1)
        has_next_calendar_month = next_month.eq(work["_vacancy_month_period"] + 1)
        work[TARGET_COLUMN] = next_target.where(has_next_calendar_month)
        work[TARGET_MONTH_COLUMN] = next_month.astype("string").where(has_next_calendar_month)

        for output_column, (source_column, lag) in LAG_FEATURES.items():
            previous_month = grouped["_vacancy_month_period"].shift(lag)
            previous_value = grouped[source_column].shift(lag)
            has_expected_lag = previous_month.eq(work["_vacancy_month_period"] - lag)
            work[output_column] = previous_value.where(has_expected_lag)

        work["month"] = work["_vacancy_month_period"].dt.month.astype("int16")
        work["quarter"] = work["_vacancy_month_period"].dt.quarter.astype("int16")
        work["year"] = work["_vacancy_month_period"].dt.year.astype("int16")
        first_month = work["_vacancy_month_period"].min()
        work["month_index"] = (
            (work["_vacancy_month_period"].dt.year - first_month.year) * 12
            + (work["_vacancy_month_period"].dt.month - first_month.month)
        ).astype("int16")

        for column in ["salary_count", "vacancy_count", "data_quality_score", "median_salary_mid"]:
            work[column] = pd.to_numeric(work[column], errors="coerce")

        filters = {
            "target_not_null": work[TARGET_COLUMN].notna(),
            "median_salary_mid_not_null": work["median_salary_mid"].notna(),
            "salary_count_min_5": work["salary_count"].ge(5),
            "vacancy_count_min_10": work["vacancy_count"].ge(10),
            "data_quality_score_min_0_5": work["data_quality_score"].ge(0.5),
        }
        keep_mask = pd.Series(True, index=work.index)
        for mask in filters.values():
            keep_mask &= mask.fillna(False)

        filtered = work.loc[keep_mask].drop(columns=["_vacancy_month_period"]).reset_index(drop=True)
        report = {
            "row_count_before": int(len(work)),
            "row_count_after": int(len(filtered)),
            "dropped_rows": int(len(work) - len(filtered)),
            "segment_key_columns": ML_SEGMENT_KEY_COLUMNS,
            "target_column": TARGET_COLUMN,
            "target_month_column": TARGET_MONTH_COLUMN,
            "lag_features": {
                output: {"source_column": source, "lag_months": lag}
                for output, (source, lag) in LAG_FEATURES.items()
            },
            "filter_pass_counts": {
                name: int(mask.fillna(False).sum()) for name, mask in filters.items()
            },
            "source_dataset_distribution": _value_counts(filtered["source_dataset"])
            if "source_dataset" in filtered.columns and len(filtered)
            else {},
            "month_min": str(filtered["vacancy_month"].min()) if len(filtered) else None,
            "month_max": str(filtered["vacancy_month"].max()) if len(filtered) else None,
            "target_month_min": str(filtered[TARGET_MONTH_COLUMN].min()) if len(filtered) else None,
            "target_month_max": str(filtered[TARGET_MONTH_COLUMN].max()) if len(filtered) else None,
        }
        return MonthlyMLDatasetResult(dataframe=filtered, report=report)


class BroadMonthlySalaryMLDatasetBuilder:
    """Prepare a broad-segment monthly next-observation salary dataset."""

    salary_weighted_columns = [
        "median_salary_mid",
        "mean_salary_mid",
        "p25_salary_mid",
        "p75_salary_mid",
        "median_salary_from",
        "median_salary_to",
    ]
    vacancy_weighted_columns = [
        "salary_missing_rate",
        "remote_share",
        "full_time_share",
        "shift_share",
        "top_employer_share",
        "data_quality_score",
    ]

    def prepare_file(
        self,
        input_path: str | Path = PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
        output_path: str | Path | None = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet",
        report_path: str | Path | None = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset_report.json",
    ) -> MonthlyMLDatasetResult:
        """Load monthly segments, prepare broad monthly rows, and optionally save outputs."""
        pd = _require_pandas()
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Monthly segments parquet not found: {path}")

        df = pd.read_parquet(path)
        result = self.prepare_dataframe(df)
        result.report["input_path"] = str(path)

        saved_output = Path(output_path) if output_path is not None else None
        saved_report = Path(report_path) if report_path is not None else None

        if saved_output is not None:
            saved_output.parent.mkdir(parents=True, exist_ok=True)
            result.dataframe.to_parquet(saved_output, index=False)
        if saved_report is not None:
            saved_report.parent.mkdir(parents=True, exist_ok=True)
            saved_report.write_text(
                json.dumps(result.report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

        result.output_path = saved_output
        result.report_path = saved_report
        return result

    def prepare_dataframe(self, df: Any) -> MonthlyMLDatasetResult:
        """Prepare broad next-observed-month supervised rows."""
        pd = _require_pandas()
        required_columns = [
            *BROAD_MONTHLY_SEGMENT_KEY_COLUMNS,
            "vacancy_month",
            "country",
            "federal_district",
            "median_salary_mid",
            "salary_count",
            "vacancy_count",
            "employer_count",
            "data_quality_score",
        ]
        missing_required = [column for column in required_columns if column not in df.columns]
        if missing_required:
            raise KeyError(f"Missing required broad monthly ML columns: {missing_required}")

        monthly = df.dropna(subset=["vacancy_month"]).copy()
        monthly["_vacancy_month_period"] = pd.PeriodIndex(
            monthly["vacancy_month"].astype("string"),
            freq="M",
        )

        for column in ["salary_count", "vacancy_count", "employer_count"]:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

        group_columns = [*BROAD_MONTHLY_SEGMENT_KEY_COLUMNS, "_vacancy_month_period"]
        grouped_monthly = monthly.groupby(group_columns, dropna=False)
        work = grouped_monthly.agg(
            country=("country", _first_non_null),
            federal_district=("federal_district", _first_non_null),
            vacancy_count=("vacancy_count", "sum"),
            salary_count=("salary_count", "sum"),
            employer_count=("employer_count", "sum"),
        ).reset_index()

        weighted_specs = [
            *[(column, "salary_count") for column in self.salary_weighted_columns if column in monthly.columns],
            *[(column, "vacancy_count") for column in self.vacancy_weighted_columns if column in monthly.columns],
        ]
        for value_column, weight_column in weighted_specs:
            values = pd.to_numeric(monthly[value_column], errors="coerce")
            weights = pd.to_numeric(monthly[weight_column], errors="coerce").fillna(0)
            valid = values.notna() & weights.gt(0)
            monthly[f"_{value_column}_weighted_sum"] = (values * weights).where(valid, 0)
            monthly[f"_{value_column}_weight_sum"] = weights.where(valid, 0)
            weighted = monthly.groupby(group_columns, dropna=False).agg(
                weighted_sum=(f"_{value_column}_weighted_sum", "sum"),
                weight_sum=(f"_{value_column}_weight_sum", "sum"),
                fallback_mean=(value_column, "mean"),
            )
            weighted_value = weighted["weighted_sum"] / weighted["weight_sum"]
            weighted_value = weighted_value.where(weighted["weight_sum"].gt(0), weighted["fallback_mean"])
            work = work.merge(
                weighted_value.rename(value_column).reset_index(),
                on=group_columns,
                how="left",
            )

        if work.empty:
            report = self._build_report(monthly, work, work, {}, {})
            return MonthlyMLDatasetResult(dataframe=work, report=report)

        work = work.sort_values(group_columns).reset_index(drop=True)
        grouped = work.groupby(BROAD_MONTHLY_SEGMENT_KEY_COLUMNS, dropna=False)

        next_month = grouped["_vacancy_month_period"].shift(-1)
        previous_month = grouped["_vacancy_month_period"].shift(1)
        work[BROAD_MONTHLY_TARGET_COLUMN] = grouped["median_salary_mid"].shift(-1)
        work[TARGET_MONTH_COLUMN] = next_month.astype("string")
        work[GAP_TO_TARGET_MONTHS_COLUMN] = self._month_gap(
            work["_vacancy_month_period"],
            next_month,
        )
        work["gap_from_previous_observed_month"] = self._month_gap(
            previous_month,
            work["_vacancy_month_period"],
        )

        for output_column, (source_column, lag) in BROAD_MONTHLY_LAG_FEATURES.items():
            work[output_column] = grouped[source_column].shift(lag)
        self._add_broad_monthly_dynamic_features(work, grouped)

        work["vacancy_month"] = work["_vacancy_month_period"].astype("string")
        work["month"] = work["_vacancy_month_period"].dt.month.astype("int16")
        work["quarter"] = work["_vacancy_month_period"].dt.quarter.astype("int16")
        work["year"] = work["_vacancy_month_period"].dt.year.astype("int16")
        first_month = work["_vacancy_month_period"].min()
        work["month_index"] = (
            (work["_vacancy_month_period"].dt.year - first_month.year) * 12
            + (work["_vacancy_month_period"].dt.month - first_month.month)
        ).astype("int16")

        for column in ["salary_count", "vacancy_count", "data_quality_score", "median_salary_mid"]:
            work[column] = pd.to_numeric(work[column], errors="coerce")

        filters = {
            "target_not_null": work[BROAD_MONTHLY_TARGET_COLUMN].notna(),
            "median_salary_mid_not_null": work["median_salary_mid"].notna(),
            "salary_count_min_1": work["salary_count"].ge(1),
            "vacancy_count_min_2": work["vacancy_count"].ge(2),
            "data_quality_score_min_0_2": work["data_quality_score"].ge(0.2),
        }
        keep_mask = pd.Series(True, index=work.index)
        sequential_filter_pass_counts = {}
        for name, mask in filters.items():
            keep_mask &= mask.fillna(False)
            sequential_filter_pass_counts[name] = int(keep_mask.sum())

        filtered = (
            work.loc[keep_mask]
            .drop(columns=["_vacancy_month_period"])
            .reset_index(drop=True)
        )
        self._add_broad_monthly_support_features(filtered)
        report = self._build_report(
            monthly,
            work,
            filtered,
            filters,
            sequential_filter_pass_counts,
        )
        return MonthlyMLDatasetResult(dataframe=filtered, report=report)

    def _month_gap(self, start: Any, end: Any) -> Any:
        pd = _require_pandas()
        gap = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
        return pd.to_numeric(gap, errors="coerce").astype("Int16")

    def _build_report(
        self,
        monthly: Any,
        broad_monthly: Any,
        filtered: Any,
        filters: dict[str, Any],
        sequential_filter_pass_counts: dict[str, int],
    ) -> dict[str, Any]:
        filter_pass_counts = {
            name: int(mask.fillna(False).sum()) for name, mask in filters.items()
        }
        return {
            "monthly_row_count_before_broad_aggregation": int(len(monthly)),
            "broad_monthly_row_count_before_filters": int(len(broad_monthly)),
            "row_count_after": int(len(filtered)),
            "dropped_rows_after_broad_aggregation": int(len(broad_monthly) - len(filtered)),
            "segment_key_columns": BROAD_MONTHLY_SEGMENT_KEY_COLUMNS,
            "target_column": BROAD_MONTHLY_TARGET_COLUMN,
            "target_month_column": TARGET_MONTH_COLUMN,
            "gap_to_target_column": GAP_TO_TARGET_MONTHS_COLUMN,
            "lag_features": {
                output: {"source_column": source, "lag_observed_periods": lag}
                for output, (source, lag) in BROAD_MONTHLY_LAG_FEATURES.items()
            },
            "dynamic_features": BROAD_MONTHLY_DYNAMIC_FEATURES,
            "dynamic_feature_missing_rates": {
                column: float(filtered[column].isna().mean())
                for column in BROAD_MONTHLY_DYNAMIC_FEATURES
                if column in filtered.columns and len(filtered)
            },
            "support_features": BROAD_MONTHLY_SUPPORT_FEATURES,
            "support_grouping": {
                "region_rare_threshold": BROAD_MONTHLY_REGION_RARE_THRESHOLD,
                "occupation_rare_threshold": BROAD_MONTHLY_OCCUPATION_RARE_THRESHOLD,
                "region_grouped_distribution": _value_counts(filtered["region_grouped"])
                if "region_grouped" in filtered.columns and len(filtered)
                else {},
                "occupation_group_grouped_distribution": _value_counts(filtered["occupation_group_grouped"])
                if "occupation_group_grouped" in filtered.columns and len(filtered)
                else {},
                "region_support_bucket_distribution": _value_counts(filtered["region_support_bucket"])
                if "region_support_bucket" in filtered.columns and len(filtered)
                else {},
                "occupation_group_support_bucket_distribution": _value_counts(filtered["occupation_group_support_bucket"])
                if "occupation_group_support_bucket" in filtered.columns and len(filtered)
                else {},
                "segment_support_bucket_distribution": _value_counts(filtered["segment_support_bucket"])
                if "segment_support_bucket" in filtered.columns and len(filtered)
                else {},
            },
            "filter_pass_counts": filter_pass_counts,
            "sequential_filter_pass_counts": sequential_filter_pass_counts,
            "source_dataset_distribution": _value_counts(filtered["source_dataset"])
            if "source_dataset" in filtered.columns and len(filtered)
            else {},
            "month_min": str(filtered["vacancy_month"].min()) if len(filtered) else None,
            "month_max": str(filtered["vacancy_month"].max()) if len(filtered) else None,
            "target_month_min": str(filtered[TARGET_MONTH_COLUMN].min()) if len(filtered) else None,
            "target_month_max": str(filtered[TARGET_MONTH_COLUMN].max()) if len(filtered) else None,
            "gap_to_target_months_distribution": _value_counts(filtered[GAP_TO_TARGET_MONTHS_COLUMN])
            if GAP_TO_TARGET_MONTHS_COLUMN in filtered.columns and len(filtered)
            else {},
        }

    def _add_broad_monthly_dynamic_features(self, work: Any, grouped: Any) -> None:
        pd = _require_pandas()
        salary = pd.to_numeric(work["median_salary_mid"], errors="coerce")
        salary_lag_1 = pd.to_numeric(work["median_salary_mid_lag_1_observed"], errors="coerce")
        salary_lag_2 = pd.to_numeric(work["median_salary_mid_lag_2_observed"], errors="coerce")
        work["median_salary_mid_delta_lag_1"] = salary - salary_lag_1
        work["median_salary_mid_pct_change_lag_1"] = self._safe_pct_change(
            salary,
            salary_lag_1,
        )
        work["median_salary_mid_lag_1_to_lag_2_delta"] = salary_lag_1 - salary_lag_2
        work["median_salary_mid_lag_1_to_lag_2_pct_change"] = self._safe_pct_change(
            salary_lag_1,
            salary_lag_2,
        )

        work["segment_previous_observation_count"] = grouped.cumcount().astype("int16")
        work["segment_previous_salary_mean"] = grouped["median_salary_mid"].transform(
            lambda series: series.expanding().mean().shift(1)
        )
        work["segment_previous_salary_median"] = grouped["median_salary_mid"].transform(
            lambda series: series.expanding().median().shift(1)
        )
        work["segment_previous_salary_std"] = grouped["median_salary_mid"].transform(
            lambda series: series.expanding().std().shift(1)
        )

        for source_column, lag_column in [
            ("vacancy_count", "vacancy_count_lag_1_observed"),
            ("salary_count", "salary_count_lag_1_observed"),
            ("data_quality_score", "data_quality_score_lag_1_observed"),
        ]:
            current = pd.to_numeric(work[source_column], errors="coerce")
            lag = pd.to_numeric(work[lag_column], errors="coerce")
            work[f"{source_column}_delta_lag_1"] = current - lag
            work[f"{source_column}_pct_change_lag_1"] = self._safe_pct_change(current, lag)

        current_employers = pd.to_numeric(work["employer_count"], errors="coerce")
        employer_lag_1 = grouped["employer_count"].shift(1)
        work["employer_count_delta_lag_1"] = current_employers - pd.to_numeric(employer_lag_1, errors="coerce")
        work["employer_count_pct_change_lag_1"] = self._safe_pct_change(
            current_employers,
            pd.to_numeric(employer_lag_1, errors="coerce"),
        )

    def _safe_pct_change(self, current: Any, previous: Any) -> Any:
        pd = _require_pandas()
        current_values = pd.to_numeric(current, errors="coerce")
        previous_values = pd.to_numeric(previous, errors="coerce")
        return (current_values - previous_values).div(previous_values.abs()).where(previous_values.ne(0))


class BroadQuarterlyModelingSetup:
    """Define features, time split, and baseline for broad quarterly salary modeling."""

    categorical_features = BROAD_QUARTERLY_CATEGORICAL_FEATURES
    numeric_features = BROAD_QUARTERLY_NUMERIC_FEATURES
    target_column = BROAD_QUARTERLY_TARGET_COLUMN
    residual_target_column = BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN
    period_column = "vacancy_quarter"

    def prepare_file(
        self,
        input_path: str | Path = PROCESSED_DATA_DIR / "ml_broad_quarterly_salary_dataset.parquet",
        output_path: str | Path | None = PROCESSED_DATA_DIR / "broad_quarterly_modeling_setup.json",
    ) -> dict[str, Any]:
        """Load a broad quarterly ML dataset, compute split and baseline, and optionally save JSON."""
        pd = _require_pandas()
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Broad quarterly ML parquet not found: {path}")

        df = pd.read_parquet(path)
        report = self.prepare_dataframe(df)
        report["input_path"] = str(path)

        saved_output = Path(output_path) if output_path is not None else None
        if saved_output is not None:
            saved_output.parent.mkdir(parents=True, exist_ok=True)
            saved_output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            report["output_path"] = str(saved_output)
        return report

    def prepare_dataframe(self, df: Any) -> dict[str, Any]:
        """Compute feature availability, time split, and naive baseline metrics."""
        pd = _require_pandas()
        required = [
            self.period_column,
            self.target_column,
            self.residual_target_column,
            "median_salary_mid",
            GAP_TO_TARGET_QUARTERS_COLUMN,
            *self.categorical_features,
        ]
        missing_required = [column for column in required if column not in df.columns]
        if missing_required:
            raise KeyError(f"Missing required broad quarterly modeling columns: {missing_required}")

        available_categorical = [column for column in self.categorical_features if column in df.columns]
        available_numeric = [column for column in self.numeric_features if column in df.columns]
        feature_columns = [*available_numeric, *available_categorical]
        missing_numeric = [column for column in self.numeric_features if column not in df.columns]

        work = df.copy()
        work["_period"] = pd.PeriodIndex(work[self.period_column].astype("string"), freq="Q")
        periods = sorted(work["_period"].dropna().unique())
        if len(periods) < 3:
            raise ValueError("Need at least three quarters for train/validation/test split.")

        validation_period = periods[-2]
        test_period = periods[-1]
        train_periods = periods[:-2]
        split_masks = {
            "train": work["_period"].isin(train_periods),
            "validation": work["_period"].eq(validation_period),
            "test": work["_period"].eq(test_period),
        }
        baseline_metrics = {
            split: self._baseline_metrics(work.loc[mask])
            for split, mask in split_masks.items()
        }
        gap_one_metrics = {
            split: self._baseline_metrics(
                work.loc[mask & work[GAP_TO_TARGET_QUARTERS_COLUMN].eq(1)]
            )
            for split, mask in split_masks.items()
        }

        return {
            "dataset_row_count": int(len(work)),
            "target_column": self.target_column,
            "residual_target_column": self.residual_target_column,
            "period_column": self.period_column,
            "baseline_prediction": "median_salary_mid",
            "categorical_features": available_categorical,
            "numeric_features": available_numeric,
            "feature_columns": feature_columns,
            "missing_numeric_features": missing_numeric,
            "catboost_cat_features": available_categorical,
            "split": {
                "train": {
                    "period_min": str(min(train_periods)),
                    "period_max": str(max(train_periods)),
                    "row_count": int(split_masks["train"].sum()),
                },
                "validation": {
                    "period": str(validation_period),
                    "row_count": int(split_masks["validation"].sum()),
                },
                "test": {
                    "period": str(test_period),
                    "row_count": int(split_masks["test"].sum()),
                },
            },
            "baseline_metrics": baseline_metrics,
            "baseline_metrics_gap_to_target_quarters_1": gap_one_metrics,
            "gap_to_target_quarters_distribution": _value_counts(work[GAP_TO_TARGET_QUARTERS_COLUMN]),
            "source_dataset_distribution": _value_counts(work["source_dataset"])
            if "source_dataset" in work.columns
            else {},
        }

    def _baseline_metrics(self, frame: Any) -> dict[str, Any]:
        pd = _require_pandas()
        if frame.empty:
            return {"row_count": 0, "valid_row_count": 0}
        actual = pd.to_numeric(frame[self.target_column], errors="coerce")
        predicted = pd.to_numeric(frame["median_salary_mid"], errors="coerce")
        valid = actual.notna() & predicted.notna()
        actual = actual[valid]
        predicted = predicted[valid]
        if len(actual) == 0:
            return {"row_count": int(len(frame)), "valid_row_count": 0}
        error = predicted - actual
        absolute_error = error.abs()
        mape = absolute_error.div(actual.abs()).replace([float("inf")], pd.NA).dropna()
        smape_denominator = actual.abs() + predicted.abs()
        smape = (2 * absolute_error).div(smape_denominator).replace([float("inf")], pd.NA).dropna()
        actual_sum = actual.abs().sum()
        return {
            "row_count": int(len(frame)),
            "valid_row_count": int(len(actual)),
            "mae": float(absolute_error.mean()),
            "medae": float(absolute_error.median()),
            "rmse": float((error.pow(2).mean()) ** 0.5),
            "mape": float(mape.mean() * 100) if len(mape) else None,
            "smape": float(smape.mean() * 100) if len(smape) else None,
            "wape": float(absolute_error.sum() / actual_sum * 100) if actual_sum else None,
            "ape_p90": float(mape.quantile(0.90) * 100) if len(mape) else None,
            "bias_mean_error": float(error.mean()),
        }

    def _add_broad_monthly_support_features(self, work: Any) -> None:
        observation_mask = work[BROAD_MONTHLY_TARGET_COLUMN].notna()
        work["_supervised_observation"] = observation_mask.astype("int16")
        work["region_observation_count"] = (
            work.groupby("region", dropna=False)["_supervised_observation"].transform("sum").astype("int32")
        )
        work["occupation_group_observation_count"] = (
            work.groupby("occupation_group", dropna=False)["_supervised_observation"].transform("sum").astype("int32")
        )
        work["segment_observation_count"] = (
            work.groupby(BROAD_MONTHLY_SEGMENT_KEY_COLUMNS, dropna=False)["_supervised_observation"]
            .transform("sum")
            .astype("int32")
        )

        work["region_grouped"] = work["region"].where(
            work["region_observation_count"].ge(BROAD_MONTHLY_REGION_RARE_THRESHOLD),
            "rare_region",
        )
        work["occupation_group_grouped"] = work["occupation_group"].where(
            work["occupation_group_observation_count"].ge(BROAD_MONTHLY_OCCUPATION_RARE_THRESHOLD),
            "rare_occupation",
        )
        work["region_support_bucket"] = self._support_bucket(work["region_observation_count"])
        work["occupation_group_support_bucket"] = self._support_bucket(work["occupation_group_observation_count"])
        work["segment_support_bucket"] = self._support_bucket(work["segment_observation_count"])
        work.drop(columns=["_supervised_observation"], inplace=True)

    def _support_bucket(self, values: Any) -> Any:
        pd = _require_pandas()
        return pd.cut(
            pd.to_numeric(values, errors="coerce").fillna(0),
            bins=[-1, 10, 30, 60, 150, float("inf")],
            labels=["very_low", "low", "medium", "high", "very_high"],
        ).astype("string")


class BroadMonthlyModelingSetup:
    """Define features, time split, and baseline for broad monthly salary modeling."""

    categorical_features = BROAD_MONTHLY_CATEGORICAL_FEATURES
    numeric_features = BROAD_MONTHLY_NUMERIC_FEATURES
    target_column = BROAD_MONTHLY_TARGET_COLUMN
    period_column = "vacancy_month"
    baseline_prediction_column = "prediction_baseline"

    def prepare_file(
        self,
        input_path: str | Path = PROCESSED_DATA_DIR / "ml_broad_monthly_salary_dataset.parquet",
        output_path: str | Path | None = PROCESSED_DATA_DIR / "broad_monthly_modeling_setup.json",
    ) -> dict[str, Any]:
        """Load a broad monthly ML dataset, compute split and baseline, and optionally save JSON."""
        pd = _require_pandas()
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Broad monthly ML parquet not found: {path}")

        df = pd.read_parquet(path)
        report = self.prepare_dataframe(df)
        report["input_path"] = str(path)

        saved_output = Path(output_path) if output_path is not None else None
        if saved_output is not None:
            saved_output.parent.mkdir(parents=True, exist_ok=True)
            saved_output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            report["output_path"] = str(saved_output)
        return report

    def prepare_dataframe(self, df: Any) -> dict[str, Any]:
        """Compute feature availability, time split, and naive baseline metrics."""
        pd = _require_pandas()
        required = [
            self.period_column,
            self.target_column,
            "median_salary_mid",
            GAP_TO_TARGET_MONTHS_COLUMN,
            *self.categorical_features,
        ]
        missing_required = [column for column in required if column not in df.columns]
        if missing_required:
            raise KeyError(f"Missing required broad monthly modeling columns: {missing_required}")

        available_categorical = [column for column in self.categorical_features if column in df.columns]
        available_numeric = [column for column in self.numeric_features if column in df.columns]
        feature_columns = [*available_numeric, *available_categorical]
        missing_numeric = [column for column in self.numeric_features if column not in df.columns]

        work = df.copy()
        work["_period"] = pd.PeriodIndex(work[self.period_column].astype("string"), freq="M")
        periods = sorted(work["_period"].dropna().unique())
        if len(periods) < 3:
            raise ValueError("Need at least three months for train/validation/test split.")

        validation_period = periods[-2]
        test_period = periods[-1]
        train_periods = periods[:-2]

        split_masks = {
            "train": work["_period"].isin(train_periods),
            "validation": work["_period"].eq(validation_period),
            "test": work["_period"].eq(test_period),
        }

        baseline_metrics = {
            split: self._baseline_metrics(work.loc[mask])
            for split, mask in split_masks.items()
        }
        gap_one_metrics = {
            split: self._baseline_metrics(
                work.loc[mask & work[GAP_TO_TARGET_MONTHS_COLUMN].eq(1)]
            )
            for split, mask in split_masks.items()
        }

        return {
            "dataset_row_count": int(len(work)),
            "target_column": self.target_column,
            "period_column": self.period_column,
            "baseline_prediction": "median_salary_mid",
            "categorical_features": available_categorical,
            "numeric_features": available_numeric,
            "feature_columns": feature_columns,
            "missing_numeric_features": missing_numeric,
            "catboost_cat_features": available_categorical,
            "split": {
                "train": {
                    "period_min": str(min(train_periods)),
                    "period_max": str(max(train_periods)),
                    "row_count": int(split_masks["train"].sum()),
                },
                "validation": {
                    "period": str(validation_period),
                    "row_count": int(split_masks["validation"].sum()),
                },
                "test": {
                    "period": str(test_period),
                    "row_count": int(split_masks["test"].sum()),
                },
            },
            "baseline_metrics": baseline_metrics,
            "baseline_metrics_gap_to_target_months_1": gap_one_metrics,
            "gap_to_target_months_distribution": _value_counts(work[GAP_TO_TARGET_MONTHS_COLUMN]),
            "source_dataset_distribution": _value_counts(work["source_dataset"]),
        }

    def _baseline_metrics(self, frame: Any) -> dict[str, Any]:
        pd = _require_pandas()
        if len(frame) == 0:
            return {"row_count": 0}

        actual = pd.to_numeric(frame[self.target_column], errors="coerce")
        predicted = pd.to_numeric(frame["median_salary_mid"], errors="coerce")
        valid = actual.notna() & predicted.notna()
        actual = actual[valid]
        predicted = predicted[valid]
        if len(actual) == 0:
            return {"row_count": int(len(frame)), "valid_row_count": 0}

        error = predicted - actual
        absolute_error = error.abs()
        percentage_error = absolute_error / actual.abs()
        smape = 2 * absolute_error / (actual.abs() + predicted.abs())

        return {
            "row_count": int(len(frame)),
            "valid_row_count": int(len(actual)),
            "mae": float(absolute_error.mean()),
            "rmse": float((error.pow(2).mean()) ** 0.5),
            "mape": float(percentage_error.mean() * 100),
            "smape": float(smape.mean() * 100),
        }


class QuarterlySalaryMLDatasetBuilder:
    """Prepare a supervised t+1 quarterly salary forecasting dataset."""

    salary_weighted_columns = [
        "median_salary_mid",
        "mean_salary_mid",
        "p25_salary_mid",
        "p75_salary_mid",
        "median_salary_from",
        "median_salary_to",
    ]
    vacancy_weighted_columns = [
        "salary_missing_rate",
        "remote_share",
        "full_time_share",
        "shift_share",
        "top_employer_share",
        "data_quality_score",
    ]

    def prepare_file(
        self,
        input_path: str | Path = PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
        output_path: str | Path | None = PROCESSED_DATA_DIR / "ml_quarterly_salary_dataset.parquet",
        report_path: str | Path | None = PROCESSED_DATA_DIR / "ml_quarterly_salary_dataset_report.json",
    ) -> QuarterlyMLDatasetResult:
        """Load monthly segments, prepare quarterly supervised rows, and optionally save outputs."""
        pd = _require_pandas()
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Monthly segments parquet not found: {path}")

        df = pd.read_parquet(path)
        result = self.prepare_dataframe(df)
        result.report["input_path"] = str(path)

        saved_output = Path(output_path) if output_path is not None else None
        saved_report = Path(report_path) if report_path is not None else None

        if saved_output is not None:
            saved_output.parent.mkdir(parents=True, exist_ok=True)
            result.dataframe.to_parquet(saved_output, index=False)
        if saved_report is not None:
            saved_report.parent.mkdir(parents=True, exist_ok=True)
            saved_report.write_text(
                json.dumps(result.report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

        result.output_path = saved_output
        result.report_path = saved_report
        return result

    def prepare_dataframe(self, df: Any) -> QuarterlyMLDatasetResult:
        """Prepare supervised rows from canonical monthly segments aggregated to quarters."""
        pd = _require_pandas()
        required_columns = [
            *ML_SEGMENT_KEY_COLUMNS,
            "vacancy_month",
            "federal_district",
            "median_salary_mid",
            "salary_count",
            "vacancy_count",
            "employer_count",
            "data_quality_score",
        ]
        missing_required = [column for column in required_columns if column not in df.columns]
        if missing_required:
            raise KeyError(f"Missing required quarterly ML columns: {missing_required}")

        monthly = df.dropna(subset=["vacancy_month"]).copy()
        monthly["_vacancy_month_period"] = pd.PeriodIndex(
            monthly["vacancy_month"].astype("string"),
            freq="M",
        )
        monthly["_vacancy_quarter_period"] = monthly["_vacancy_month_period"].dt.asfreq("Q")

        for column in ["salary_count", "vacancy_count", "employer_count"]:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

        group_columns = [*ML_SEGMENT_KEY_COLUMNS, "_vacancy_quarter_period"]
        grouped_monthly = monthly.groupby(group_columns, dropna=False)
        work = grouped_monthly.agg(
            federal_district=("federal_district", _first_non_null),
            vacancy_count=("vacancy_count", "sum"),
            salary_count=("salary_count", "sum"),
            employer_count=("employer_count", "sum"),
            months_observed_in_quarter=("_vacancy_month_period", "nunique"),
        ).reset_index()

        weighted_specs = [
            *[(column, "salary_count") for column in self.salary_weighted_columns if column in monthly.columns],
            *[(column, "vacancy_count") for column in self.vacancy_weighted_columns if column in monthly.columns],
        ]
        for value_column, weight_column in weighted_specs:
            values = pd.to_numeric(monthly[value_column], errors="coerce")
            weights = pd.to_numeric(monthly[weight_column], errors="coerce").fillna(0)
            valid = values.notna() & weights.gt(0)
            monthly[f"_{value_column}_weighted_sum"] = (values * weights).where(valid, 0)
            monthly[f"_{value_column}_weight_sum"] = weights.where(valid, 0)
            weighted = monthly.groupby(group_columns, dropna=False).agg(
                weighted_sum=(f"_{value_column}_weighted_sum", "sum"),
                weight_sum=(f"_{value_column}_weight_sum", "sum"),
                fallback_mean=(value_column, "mean"),
            )
            weighted_value = weighted["weighted_sum"] / weighted["weight_sum"]
            weighted_value = weighted_value.where(weighted["weight_sum"].gt(0), weighted["fallback_mean"])
            work = work.merge(
                weighted_value.rename(value_column).reset_index(),
                on=group_columns,
                how="left",
            )
        if work.empty:
            report = self._build_report(monthly, work, work, {})
            return QuarterlyMLDatasetResult(dataframe=work, report=report)

        work = work.sort_values(group_columns).reset_index(drop=True)
        grouped = work.groupby(ML_SEGMENT_KEY_COLUMNS, dropna=False)

        next_quarter = grouped["_vacancy_quarter_period"].shift(-1)
        next_target = grouped["median_salary_mid"].shift(-1)
        has_next_calendar_quarter = next_quarter.eq(work["_vacancy_quarter_period"] + 1)
        work[QUARTERLY_TARGET_COLUMN] = next_target.where(has_next_calendar_quarter)
        work[TARGET_QUARTER_COLUMN] = next_quarter.astype("string").where(has_next_calendar_quarter)

        for output_column, (source_column, lag) in QUARTERLY_LAG_FEATURES.items():
            previous_quarter = grouped["_vacancy_quarter_period"].shift(lag)
            previous_value = grouped[source_column].shift(lag)
            has_expected_lag = previous_quarter.eq(work["_vacancy_quarter_period"] - lag)
            work[output_column] = previous_value.where(has_expected_lag)

        work["vacancy_quarter"] = work["_vacancy_quarter_period"].astype("string")
        work["quarter"] = work["_vacancy_quarter_period"].dt.quarter.astype("int16")
        work["year"] = work["_vacancy_quarter_period"].dt.year.astype("int16")
        first_quarter = work["_vacancy_quarter_period"].min()
        work["quarter_index"] = (
            (work["_vacancy_quarter_period"].dt.year - first_quarter.year) * 4
            + (work["_vacancy_quarter_period"].dt.quarter - first_quarter.quarter)
        ).astype("int16")
        numeric_filter_columns = [
            "salary_count",
            "vacancy_count",
            "data_quality_score",
            "median_salary_mid",
            "months_observed_in_quarter",
        ]
        for column in numeric_filter_columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

        filters = {
            "target_not_null": work[QUARTERLY_TARGET_COLUMN].notna(),
            "median_salary_mid_not_null": work["median_salary_mid"].notna(),
            "months_observed_in_quarter_min_1": work["months_observed_in_quarter"].ge(1),
            "salary_count_min_2": work["salary_count"].ge(2),
            "vacancy_count_min_3": work["vacancy_count"].ge(3),
            "data_quality_score_min_0_3": work["data_quality_score"].ge(0.3),
        }
        keep_mask = pd.Series(True, index=work.index)
        sequential_filter_pass_counts = {}
        for name, mask in filters.items():
            keep_mask &= mask.fillna(False)
            sequential_filter_pass_counts[name] = int(keep_mask.sum())

        filtered = (
            work.loc[keep_mask]
            .drop(columns=["_vacancy_quarter_period"])
            .reset_index(drop=True)
        )
        report = self._build_report(monthly, work, filtered, filters)
        report["sequential_filter_pass_counts"] = sequential_filter_pass_counts
        return QuarterlyMLDatasetResult(dataframe=filtered, report=report)

    def _build_report(
        self,
        monthly: Any,
        quarterly: Any,
        filtered: Any,
        filters: dict[str, Any],
    ) -> dict[str, Any]:
        filter_pass_counts = {
            name: int(mask.fillna(False).sum()) for name, mask in filters.items()
        }
        return {
            "monthly_row_count_before_quarterly_aggregation": int(len(monthly)),
            "quarterly_row_count_before_filters": int(len(quarterly)),
            "row_count_after": int(len(filtered)),
            "dropped_rows_after_quarterly_aggregation": int(len(quarterly) - len(filtered)),
            "segment_key_columns": ML_SEGMENT_KEY_COLUMNS,
            "target_column": QUARTERLY_TARGET_COLUMN,
            "target_quarter_column": TARGET_QUARTER_COLUMN,
            "lag_features": {
                output: {"source_column": source, "lag_quarters": lag}
                for output, (source, lag) in QUARTERLY_LAG_FEATURES.items()
            },
            "filter_pass_counts": filter_pass_counts,
            "source_dataset_distribution": _value_counts(filtered["source_dataset"])
            if "source_dataset" in filtered.columns and len(filtered)
            else {},
            "quarter_min": str(filtered["vacancy_quarter"].min()) if len(filtered) else None,
            "quarter_max": str(filtered["vacancy_quarter"].max()) if len(filtered) else None,
            "target_quarter_min": str(filtered[TARGET_QUARTER_COLUMN].min()) if len(filtered) else None,
            "target_quarter_max": str(filtered[TARGET_QUARTER_COLUMN].max()) if len(filtered) else None,
            "months_observed_distribution": _value_counts(filtered["months_observed_in_quarter"])
            if "months_observed_in_quarter" in filtered.columns and len(filtered)
            else {},
        }


class BroadQuarterlySalaryMLDatasetBuilder:
    """Prepare a broad-segment quarterly next-observation salary dataset."""

    salary_weighted_columns = QuarterlySalaryMLDatasetBuilder.salary_weighted_columns
    vacancy_weighted_columns = QuarterlySalaryMLDatasetBuilder.vacancy_weighted_columns

    def prepare_file(
        self,
        input_path: str | Path = PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
        output_path: str | Path | None = PROCESSED_DATA_DIR / "ml_broad_quarterly_salary_dataset.parquet",
        report_path: str | Path | None = PROCESSED_DATA_DIR / "ml_broad_quarterly_salary_dataset_report.json",
    ) -> QuarterlyMLDatasetResult:
        """Load monthly segments, prepare broad quarterly rows, and optionally save outputs."""
        pd = _require_pandas()
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Monthly segments parquet not found: {path}")

        df = pd.read_parquet(path)
        result = self.prepare_dataframe(df)
        result.report["input_path"] = str(path)

        saved_output = Path(output_path) if output_path is not None else None
        saved_report = Path(report_path) if report_path is not None else None

        if saved_output is not None:
            saved_output.parent.mkdir(parents=True, exist_ok=True)
            result.dataframe.to_parquet(saved_output, index=False)
        if saved_report is not None:
            saved_report.parent.mkdir(parents=True, exist_ok=True)
            saved_report.write_text(
                json.dumps(result.report, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

        result.output_path = saved_output
        result.report_path = saved_report
        return result

    def prepare_dataframe(self, df: Any) -> QuarterlyMLDatasetResult:
        """Prepare broad next-observed-quarter supervised rows."""
        pd = _require_pandas()
        required_columns = [
            *BROAD_QUARTERLY_SEGMENT_KEY_COLUMNS,
            "vacancy_month",
            "country",
            "federal_district",
            "median_salary_mid",
            "salary_count",
            "vacancy_count",
            "employer_count",
            "data_quality_score",
        ]
        missing_required = [column for column in required_columns if column not in df.columns]
        if missing_required:
            raise KeyError(f"Missing required broad quarterly ML columns: {missing_required}")

        monthly = df.dropna(subset=["vacancy_month"]).copy()
        monthly["_vacancy_month_period"] = pd.PeriodIndex(
            monthly["vacancy_month"].astype("string"),
            freq="M",
        )
        monthly["_vacancy_quarter_period"] = monthly["_vacancy_month_period"].dt.asfreq("Q")

        for column in ["salary_count", "vacancy_count", "employer_count"]:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

        group_columns = [*BROAD_QUARTERLY_SEGMENT_KEY_COLUMNS, "_vacancy_quarter_period"]
        grouped_monthly = monthly.groupby(group_columns, dropna=False)
        work = grouped_monthly.agg(
            country=("country", _first_non_null),
            federal_district=("federal_district", _first_non_null),
            vacancy_count=("vacancy_count", "sum"),
            salary_count=("salary_count", "sum"),
            employer_count=("employer_count", "sum"),
            months_observed_in_quarter=("_vacancy_month_period", "nunique"),
        ).reset_index()

        weighted_specs = [
            *[(column, "salary_count") for column in self.salary_weighted_columns if column in monthly.columns],
            *[(column, "vacancy_count") for column in self.vacancy_weighted_columns if column in monthly.columns],
        ]
        for value_column, weight_column in weighted_specs:
            values = pd.to_numeric(monthly[value_column], errors="coerce")
            weights = pd.to_numeric(monthly[weight_column], errors="coerce").fillna(0)
            valid = values.notna() & weights.gt(0)
            monthly[f"_{value_column}_weighted_sum"] = (values * weights).where(valid, 0)
            monthly[f"_{value_column}_weight_sum"] = weights.where(valid, 0)
            weighted = monthly.groupby(group_columns, dropna=False).agg(
                weighted_sum=(f"_{value_column}_weighted_sum", "sum"),
                weight_sum=(f"_{value_column}_weight_sum", "sum"),
                fallback_mean=(value_column, "mean"),
            )
            weighted_value = weighted["weighted_sum"] / weighted["weight_sum"]
            weighted_value = weighted_value.where(weighted["weight_sum"].gt(0), weighted["fallback_mean"])
            work = work.merge(
                weighted_value.rename(value_column).reset_index(),
                on=group_columns,
                how="left",
            )

        if work.empty:
            report = self._build_report(monthly, work, work, {}, {})
            return QuarterlyMLDatasetResult(dataframe=work, report=report)

        work = work.sort_values(group_columns).reset_index(drop=True)
        grouped = work.groupby(BROAD_QUARTERLY_SEGMENT_KEY_COLUMNS, dropna=False)

        next_quarter = grouped["_vacancy_quarter_period"].shift(-1)
        previous_quarter = grouped["_vacancy_quarter_period"].shift(1)
        work[BROAD_QUARTERLY_TARGET_COLUMN] = grouped["median_salary_mid"].shift(-1)
        work[TARGET_QUARTER_COLUMN] = next_quarter.astype("string")
        work[GAP_TO_TARGET_QUARTERS_COLUMN] = self._quarter_gap(
            work["_vacancy_quarter_period"],
            next_quarter,
        )
        work["gap_from_previous_observed_quarter"] = self._quarter_gap(
            previous_quarter,
            work["_vacancy_quarter_period"],
        )

        for output_column, (source_column, lag) in BROAD_QUARTERLY_LAG_FEATURES.items():
            work[output_column] = grouped[source_column].shift(lag)
        self._add_broad_quarterly_dynamic_features(work, grouped)

        work["vacancy_quarter"] = work["_vacancy_quarter_period"].astype("string")
        work["quarter"] = work["_vacancy_quarter_period"].dt.quarter.astype("int16")
        work["year"] = work["_vacancy_quarter_period"].dt.year.astype("int16")
        first_quarter = work["_vacancy_quarter_period"].min()
        work["quarter_index"] = (
            (work["_vacancy_quarter_period"].dt.year - first_quarter.year) * 4
            + (work["_vacancy_quarter_period"].dt.quarter - first_quarter.quarter)
        ).astype("int16")
        work["target_quarter"] = next_quarter.dt.quarter.astype("Int16")
        work["target_year"] = next_quarter.dt.year.astype("Int16")
        work["target_quarter_index"] = (
            (next_quarter.dt.year - first_quarter.year) * 4
            + (next_quarter.dt.quarter - first_quarter.quarter)
        ).astype("Int16")
        work["quarter_transition"] = (
            work["quarter"].astype("string") + "_to_" + work["target_quarter"].astype("string")
        )
        work["quarter_transition"] = work["quarter_transition"].where(next_quarter.notna())
        work["is_q4_to_q1"] = (work["quarter"].eq(4) & work["target_quarter"].eq(1)).astype("int8")
        self._add_seasonal_transition_features(work)

        for column in [
            "salary_count",
            "vacancy_count",
            "data_quality_score",
            "median_salary_mid",
            "months_observed_in_quarter",
        ]:
            work[column] = pd.to_numeric(work[column], errors="coerce")

        filters = {
            "target_not_null": work[BROAD_QUARTERLY_TARGET_COLUMN].notna(),
            "median_salary_mid_not_null": work["median_salary_mid"].notna(),
            "salary_count_min_1": work["salary_count"].ge(1),
            "vacancy_count_min_2": work["vacancy_count"].ge(2),
            "data_quality_score_min_0_2": work["data_quality_score"].ge(0.2),
        }
        keep_mask = pd.Series(True, index=work.index)
        sequential_filter_pass_counts = {}
        for name, mask in filters.items():
            keep_mask &= mask.fillna(False)
            sequential_filter_pass_counts[name] = int(keep_mask.sum())

        filtered = (
            work.loc[keep_mask]
            .drop(columns=["_vacancy_quarter_period"])
            .reset_index(drop=True)
        )
        self._add_residual_target(filtered)
        report = self._build_report(
            monthly,
            work,
            filtered,
            filters,
            sequential_filter_pass_counts,
        )
        return QuarterlyMLDatasetResult(dataframe=filtered, report=report)

    def _add_residual_target(self, filtered: Any) -> None:
        pd = _require_pandas()
        target = pd.to_numeric(filtered[BROAD_QUARTERLY_TARGET_COLUMN], errors="coerce")
        baseline = pd.to_numeric(filtered["median_salary_mid"], errors="coerce")
        valid = target.gt(0) & baseline.gt(0)
        filtered[BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN] = None
        filtered.loc[valid, BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN] = (
            target.loc[valid].map(math.log) - baseline.loc[valid].map(math.log)
        )
        filtered[BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN] = pd.to_numeric(
            filtered[BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN],
            errors="coerce",
        )
        filtered["prediction_baseline"] = filtered["median_salary_mid"]

    def _quarter_gap(self, start: Any, end: Any) -> Any:
        pd = _require_pandas()
        gap = (end.dt.year - start.dt.year) * 4 + (end.dt.quarter - start.dt.quarter)
        return pd.to_numeric(gap, errors="coerce").astype("Int16")

    def _build_report(
        self,
        monthly: Any,
        quarterly: Any,
        filtered: Any,
        filters: dict[str, Any],
        sequential_filter_pass_counts: dict[str, int],
    ) -> dict[str, Any]:
        filter_pass_counts = {
            name: int(mask.fillna(False).sum()) for name, mask in filters.items()
        }
        return {
            "monthly_row_count_before_quarterly_aggregation": int(len(monthly)),
            "quarterly_row_count_before_filters": int(len(quarterly)),
            "row_count_after": int(len(filtered)),
            "dropped_rows_after_quarterly_aggregation": int(len(quarterly) - len(filtered)),
            "segment_key_columns": BROAD_QUARTERLY_SEGMENT_KEY_COLUMNS,
            "target_column": BROAD_QUARTERLY_TARGET_COLUMN,
            "residual_target_column": BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN,
            "target_quarter_column": TARGET_QUARTER_COLUMN,
            "gap_to_target_column": GAP_TO_TARGET_QUARTERS_COLUMN,
            "lag_features": {
                output: {"source_column": source, "lag_observed_periods": lag}
                for output, (source, lag) in BROAD_QUARTERLY_LAG_FEATURES.items()
            },
            "dynamic_features": BROAD_QUARTERLY_DYNAMIC_FEATURES,
            "dynamic_feature_missing_rates": {
                column: float(filtered[column].isna().mean())
                for column in BROAD_QUARTERLY_DYNAMIC_FEATURES
                if column in filtered.columns and len(filtered)
            },
            "filter_pass_counts": filter_pass_counts,
            "sequential_filter_pass_counts": sequential_filter_pass_counts,
            "residual_target_non_null_count": int(filtered[BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN].notna().sum())
            if BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN in filtered.columns
            else 0,
            "residual_target_missing_count": int(filtered[BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN].isna().sum())
            if BROAD_QUARTERLY_RESIDUAL_TARGET_COLUMN in filtered.columns
            else 0,
            "source_dataset_distribution": _value_counts(filtered["source_dataset"])
            if "source_dataset" in filtered.columns and len(filtered)
            else {},
            "quarter_min": str(filtered["vacancy_quarter"].min()) if len(filtered) else None,
            "quarter_max": str(filtered["vacancy_quarter"].max()) if len(filtered) else None,
            "target_quarter_min": str(filtered[TARGET_QUARTER_COLUMN].min()) if len(filtered) else None,
            "target_quarter_max": str(filtered[TARGET_QUARTER_COLUMN].max()) if len(filtered) else None,
            "gap_to_target_quarters_distribution": _value_counts(filtered[GAP_TO_TARGET_QUARTERS_COLUMN])
            if GAP_TO_TARGET_QUARTERS_COLUMN in filtered.columns and len(filtered)
            else {},
            "months_observed_distribution": _value_counts(filtered["months_observed_in_quarter"])
            if "months_observed_in_quarter" in filtered.columns and len(filtered)
            else {},
        }

    def _add_broad_quarterly_dynamic_features(self, work: Any, grouped: Any) -> None:
        pd = _require_pandas()
        salary = pd.to_numeric(work["median_salary_mid"], errors="coerce")
        salary_lag_1 = pd.to_numeric(work["median_salary_mid_lag_1_observed"], errors="coerce")
        salary_lag_2 = pd.to_numeric(work["median_salary_mid_lag_2_observed"], errors="coerce")
        work["median_salary_mid_delta_lag_1"] = salary - salary_lag_1
        work["median_salary_mid_pct_change_lag_1"] = self._safe_pct_change(salary, salary_lag_1)
        work["median_salary_mid_lag_1_to_lag_2_delta"] = salary_lag_1 - salary_lag_2
        work["median_salary_mid_lag_1_to_lag_2_pct_change"] = self._safe_pct_change(salary_lag_1, salary_lag_2)

        work["segment_previous_observation_count"] = grouped.cumcount().astype("int16")
        work["segment_previous_salary_mean"] = grouped["median_salary_mid"].transform(
            lambda series: series.expanding().mean().shift(1)
        )
        work["segment_previous_salary_median"] = grouped["median_salary_mid"].transform(
            lambda series: series.expanding().median().shift(1)
        )
        work["segment_previous_salary_std"] = grouped["median_salary_mid"].transform(
            lambda series: series.expanding().std().shift(1)
        )

        for source_column, lag_column in [
            ("vacancy_count", "vacancy_count_lag_1_observed"),
            ("salary_count", "salary_count_lag_1_observed"),
            ("data_quality_score", "data_quality_score_lag_1_observed"),
        ]:
            current = pd.to_numeric(work[source_column], errors="coerce")
            lag = pd.to_numeric(work[lag_column], errors="coerce")
            work[f"{source_column}_delta_lag_1"] = current - lag
            work[f"{source_column}_pct_change_lag_1"] = self._safe_pct_change(current, lag)

        current_employers = pd.to_numeric(work["employer_count"], errors="coerce")
        employer_lag_1 = pd.to_numeric(grouped["employer_count"].shift(1), errors="coerce")
        work["employer_count_delta_lag_1"] = current_employers - employer_lag_1
        work["employer_count_pct_change_lag_1"] = self._safe_pct_change(current_employers, employer_lag_1)

    def _add_seasonal_transition_features(self, work: Any) -> None:
        pd = _require_pandas()
        baseline = pd.to_numeric(work["median_salary_mid"], errors="coerce")
        target = pd.to_numeric(work[BROAD_QUARTERLY_TARGET_COLUMN], errors="coerce")
        valid = baseline.gt(0) & target.gt(0) & work["quarter_transition"].notna()

        history = work.loc[
            valid,
            ["_vacancy_quarter_period", "quarter_transition", "median_salary_mid", BROAD_QUARTERLY_TARGET_COLUMN],
        ].copy()
        history["salary_ratio"] = (
            pd.to_numeric(history[BROAD_QUARTERLY_TARGET_COLUMN], errors="coerce")
            / pd.to_numeric(history["median_salary_mid"], errors="coerce")
        )
        history["log_delta"] = history["salary_ratio"].map(math.log)
        history_by_period = (
            history.groupby(["quarter_transition", "_vacancy_quarter_period"], dropna=False)
            .agg(
                period_count=("salary_ratio", "size"),
                period_ratio_mean=("salary_ratio", "mean"),
                period_ratio_median=("salary_ratio", "median"),
                period_log_delta_mean=("log_delta", "mean"),
            )
            .reset_index()
            .sort_values(["quarter_transition", "_vacancy_quarter_period"])
        )
        if history_by_period.empty:
            work["seasonal_transition_previous_count"] = pd.NA
            work["seasonal_transition_salary_ratio_mean"] = pd.NA
            work["seasonal_transition_salary_ratio_median"] = pd.NA
            work["seasonal_transition_log_delta_mean"] = pd.NA
            return

        grouped_history = history_by_period.groupby("quarter_transition", dropna=False)
        previous_count = grouped_history["period_count"].cumsum() - history_by_period["period_count"]
        weighted_ratio = history_by_period["period_ratio_mean"] * history_by_period["period_count"]
        weighted_log_delta = history_by_period["period_log_delta_mean"] * history_by_period["period_count"]
        previous_ratio_sum = weighted_ratio.groupby(history_by_period["quarter_transition"], dropna=False).cumsum()
        previous_ratio_sum = previous_ratio_sum - weighted_ratio
        previous_log_delta_sum = weighted_log_delta.groupby(history_by_period["quarter_transition"], dropna=False).cumsum()
        previous_log_delta_sum = previous_log_delta_sum - weighted_log_delta
        history_by_period["seasonal_transition_previous_count"] = previous_count
        history_by_period["seasonal_transition_salary_ratio_mean"] = previous_ratio_sum.div(previous_count).where(
            previous_count.gt(0)
        )
        history_by_period["seasonal_transition_log_delta_mean"] = previous_log_delta_sum.div(previous_count).where(
            previous_count.gt(0)
        )
        history_by_period["seasonal_transition_salary_ratio_median"] = grouped_history[
            "period_ratio_median"
        ].transform(lambda series: series.expanding().median().shift(1))

        seasonal_features = history_by_period[
            [
                "quarter_transition",
                "_vacancy_quarter_period",
                "seasonal_transition_previous_count",
                "seasonal_transition_salary_ratio_mean",
                "seasonal_transition_salary_ratio_median",
                "seasonal_transition_log_delta_mean",
            ]
        ]
        work_with_features = work.merge(
            seasonal_features,
            on=["quarter_transition", "_vacancy_quarter_period"],
            how="left",
        )
        for column in [
            "seasonal_transition_previous_count",
            "seasonal_transition_salary_ratio_mean",
            "seasonal_transition_salary_ratio_median",
            "seasonal_transition_log_delta_mean",
        ]:
            work[column] = work_with_features[column]

    def _safe_pct_change(self, current: Any, previous: Any) -> Any:
        pd = _require_pandas()
        current_values = pd.to_numeric(current, errors="coerce")
        previous_values = pd.to_numeric(previous, errors="coerce")
        return (current_values - previous_values).div(previous_values.abs()).where(previous_values.ne(0))
