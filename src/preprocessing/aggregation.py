"""Aggregations for monthly segment modeling datasets."""

from pathlib import Path
from typing import Any

from .constants import PROCESSED_DATA_DIR


MONTHLY_SEGMENT_COLUMNS = [
    "source_dataset",
    "vacancy_month",
    "country",
    "region",
    "federal_district",
    "occupation_group",
    "occupation_code",
    "experience_group",
    "employment_type",
    "schedule_type",
    "vacancy_count",
    "salary_count",
    "salary_missing_rate",
    "median_salary_mid",
    "mean_salary_mid",
    "p25_salary_mid",
    "p75_salary_mid",
    "median_salary_from",
    "median_salary_to",
    "remote_share",
    "full_time_share",
    "shift_share",
    "employer_count",
    "top_employer_share",
    "data_quality_score",
]

DEFAULT_MONTHLY_GROUP_COLUMNS = [
    "source_dataset",
    "vacancy_month",
    "country",
    "region",
    "federal_district",
    "occupation_group",
    "occupation_code",
    "experience_group",
    "employment_type",
    "schedule_type",
]


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("SegmentAggregator requires pandas.") from exc
    return pd


class SegmentAggregator:
    """Aggregate canonical vacancy rows to source-aware monthly segments."""

    def aggregate_monthly_segments(
        self,
        df: Any,
        group_columns: list[str] | None = None,
    ) -> Any:
        """Build the canonical monthly segment dataset."""
        pd = _require_pandas()
        group_columns = group_columns or DEFAULT_MONTHLY_GROUP_COLUMNS
        out = df.copy()

        for column in group_columns:
            if column not in out.columns:
                out[column] = pd.NA

        for column in ["salary_mid", "salary_from", "salary_to", "quality_flag_count"]:
            if column not in out.columns:
                out[column] = pd.NA
            out[column] = pd.to_numeric(out[column], errors="coerce")

        if "is_remote" not in out.columns:
            out["is_remote"] = False
        if "is_shift" not in out.columns:
            out["is_shift"] = False
        if "schedule_type" not in out.columns:
            out["schedule_type"] = pd.NA
        if "employment_type" not in out.columns:
            out["employment_type"] = pd.NA
        if "employer_id" not in out.columns:
            out["employer_id"] = pd.NA
        if "employer_name" not in out.columns:
            out["employer_name"] = pd.NA

        out["_has_salary_mid"] = out["salary_mid"].notna()
        out["_is_remote"] = out["is_remote"].fillna(False).astype(bool)
        out["_is_shift"] = out["is_shift"].fillna(False).astype(bool)
        schedule = out["schedule_type"].astype("string").str.casefold()
        employment = out["employment_type"].astype("string").str.casefold()
        out["_is_full_time"] = schedule.eq("full_day") | employment.isin(
            ["full_time", "fulltime", "полная занятость"]
        )
        out["_is_clean_quality"] = out["quality_flag_count"].fillna(0).eq(0)

        grouped = out.groupby(group_columns, dropna=False)
        result = grouped.agg(
            vacancy_count=("source_dataset", "size"),
            salary_count=("_has_salary_mid", "sum"),
            median_salary_mid=("salary_mid", "median"),
            mean_salary_mid=("salary_mid", "mean"),
            p25_salary_mid=("salary_mid", lambda series: series.quantile(0.25)),
            p75_salary_mid=("salary_mid", lambda series: series.quantile(0.75)),
            median_salary_from=("salary_from", "median"),
            median_salary_to=("salary_to", "median"),
            remote_share=("_is_remote", "mean"),
            full_time_share=("_is_full_time", "mean"),
            shift_share=("_is_shift", "mean"),
            data_quality_score=("_is_clean_quality", "mean"),
        ).reset_index()

        result["salary_count"] = result["salary_count"].astype("int64")
        result["salary_missing_rate"] = 1 - (
            result["salary_count"] / result["vacancy_count"]
        )

        employer_key = out["employer_id"].where(out["employer_id"].notna(), out["employer_name"])
        employer_work = out[group_columns].copy()
        employer_work["_employer_key"] = employer_key
        employer_count = (
            employer_work.groupby(group_columns, dropna=False)["_employer_key"]
            .nunique(dropna=True)
            .reset_index(name="employer_count")
        )

        employer_non_null = employer_work[employer_work["_employer_key"].notna()]
        if employer_non_null.empty:
            top_share = result[group_columns].copy()
            top_share["top_employer_share"] = 0.0
        else:
            employer_sizes = (
                employer_non_null.groupby(group_columns + ["_employer_key"], dropna=False)
                .size()
                .reset_index(name="employer_vacancy_count")
            )
            top_share = (
                employer_sizes.groupby(group_columns, dropna=False)["employer_vacancy_count"]
                .max()
                .reset_index(name="top_employer_count")
            )
            top_share = top_share.merge(
                result[group_columns + ["vacancy_count"]],
                on=group_columns,
                how="left",
            )
            top_share["top_employer_share"] = (
                top_share["top_employer_count"] / top_share["vacancy_count"]
            )
            top_share = top_share[group_columns + ["top_employer_share"]]

        result = result.merge(employer_count, on=group_columns, how="left")
        result = result.merge(top_share, on=group_columns, how="left")
        result["employer_count"] = result["employer_count"].fillna(0).astype("int64")
        result["top_employer_share"] = result["top_employer_share"].fillna(0.0)

        return result.reindex(columns=MONTHLY_SEGMENT_COLUMNS)

    def compute_share(
        self,
        df: Any,
        group_columns: list[str],
        condition_column: str,
        condition_value: Any,
        output_column: str,
    ) -> Any:
        """Compute a grouped share for binary or categorical conditions."""
        pd = _require_pandas()
        missing = [column for column in [*group_columns, condition_column] if column not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for share computation: {missing}")

        work = df[group_columns + [condition_column]].copy()
        if callable(condition_value):
            work["_condition"] = work[condition_column].map(condition_value)
        else:
            work["_condition"] = work[condition_column].eq(condition_value)
        work["_condition"] = work["_condition"].fillna(False).astype(bool)

        return (
            work.groupby(group_columns, dropna=False)["_condition"]
            .mean()
            .reset_index(name=output_column)
        )

    def save_monthly_segments(
        self,
        df: Any,
        output_path: str | Path = PROCESSED_DATA_DIR / "canonical_monthly_segments.parquet",
        group_columns: list[str] | None = None,
    ) -> Any:
        """Aggregate and save canonical monthly segments to parquet."""
        segments = self.aggregate_monthly_segments(df, group_columns=group_columns)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        segments.to_parquet(path, index=False)
        return segments
