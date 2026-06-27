"""Date parsing and canonical date selection."""

from typing import Any

from .constants import (
    SOURCE_COMBINED,
    SOURCE_HH_GITHUB,
    SOURCE_HH_KAGGLE,
    SOURCE_TRUDVSEM_LATEST,
    validate_source_dataset,
)


SOURCE_DATE_CONFIG = {
    SOURCE_TRUDVSEM_LATEST: {
        "source_column": "creationDate",
        "canonical_raw_column": "created_at_raw",
        "date_type": "created_at",
    },
    SOURCE_HH_GITHUB: {
        "source_column": "published_at",
        "canonical_raw_column": "published_at_raw",
        "date_type": "published_at",
    },
    SOURCE_HH_KAGGLE: {
        "source_column": "date_of_post",
        "canonical_raw_column": "published_at_raw",
        "date_type": "published_at",
    },
    SOURCE_COMBINED: {
        "source_column": "last_found_at",
        "canonical_raw_column": "last_seen_at_raw",
        "date_type": "last_seen_at",
    },
}

DEFAULT_MIN_DATE = "2000-01-01"
DEFAULT_MAX_DATE = "2026-05-21"


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("DateProcessor requires pandas.") from exc
    return pd


def _clean_date_series(series: Any) -> Any:
    return series.astype("string").str.strip().str.strip('"').str.strip("'")


def _looks_dayfirst(series: Any) -> bool:
    sample = _clean_date_series(series).dropna().head(100)
    if sample.empty:
        return False
    return bool(sample.str.match(r"^\d{1,2}\.\d{1,2}\.\d{2,4}").any())


class DateProcessor:
    """Parse source date columns and choose a canonical vacancy date."""

    def parse_datetime_column(self, df: Any, column: str, **kwargs: Any) -> Any:
        """Parse a raw date/datetime column."""
        pd = _require_pandas()
        if column not in df.columns:
            raise KeyError(f"Column {column!r} not found in dataframe.")

        out = df.copy()
        raw_column = kwargs.pop("raw_column", f"{column}_raw")
        parsed_column = kwargs.pop("parsed_column", column)
        failed_column = kwargs.pop("failed_column", f"{column}_parse_failed")
        preserve_raw = kwargs.pop("preserve_raw", True)
        dayfirst = kwargs.pop("dayfirst", _looks_dayfirst(out[column]))

        raw_values = out[column]
        if preserve_raw and raw_column not in out.columns:
            out[raw_column] = raw_values
        parsed_raw_column = f"{parsed_column}_raw"
        if preserve_raw and parsed_column != column and parsed_raw_column not in out.columns:
            out[parsed_raw_column] = raw_values

        cleaned = _clean_date_series(raw_values)
        parsed = pd.to_datetime(
            cleaned,
            errors="coerce",
            utc=True,
            dayfirst=dayfirst,
            format=kwargs.pop("format", "mixed"),
            **kwargs,
        )
        out[parsed_column] = parsed
        out[failed_column] = raw_values.notna() & parsed.isna()
        return out

    def select_vacancy_date(self, df: Any, source_dataset: str) -> Any:
        """Create canonical vacancy date fields for a source dataset."""
        pd = _require_pandas()
        source = validate_source_dataset(source_dataset)
        config = SOURCE_DATE_CONFIG[source]
        out = df.copy()

        raw_candidates = [
            config["canonical_raw_column"],
            config["source_column"],
            "vacancy_date_raw",
        ]
        raw_column = next((column for column in raw_candidates if column in out.columns), None)
        if raw_column is None:
            raise KeyError(
                f"No date column found for {source_dataset!r}. "
                f"Checked: {raw_candidates}"
            )

        out["vacancy_date_raw"] = out[raw_column]
        parsed = pd.to_datetime(
            _clean_date_series(out["vacancy_date_raw"]),
            errors="coerce",
            utc=True,
            dayfirst=_looks_dayfirst(out["vacancy_date_raw"]),
            format="mixed",
        )
        out["vacancy_date"] = parsed
        out["vacancy_date_type"] = config["date_type"]
        out["vacancy_date_parse_failed"] = out["vacancy_date_raw"].notna() & parsed.isna()

        if source == SOURCE_TRUDVSEM_LATEST:
            snapshot_raw = out["modified_at_raw"] if "modified_at_raw" in out.columns else out["vacancy_date_raw"]
            out["snapshot_date"] = pd.to_datetime(
                _clean_date_series(snapshot_raw),
                errors="coerce",
                utc=True,
                dayfirst=_looks_dayfirst(snapshot_raw),
                format="mixed",
            )
        elif source == SOURCE_COMBINED:
            out["snapshot_date"] = out["vacancy_date"]
        elif "snapshot_date" not in out.columns:
            out["snapshot_date"] = pd.NaT

        return out

    def extract_vacancy_month(self, df: Any, date_column: str = "vacancy_date") -> Any:
        """Create a monthly period column from the canonical vacancy date."""
        pd = _require_pandas()
        if date_column not in df.columns:
            raise KeyError(f"Column {date_column!r} not found in dataframe.")

        out = df.copy()
        dates = pd.to_datetime(out[date_column], errors="coerce", utc=True)
        out["vacancy_month"] = dates.dt.tz_convert(None).dt.to_period("M").astype("string")
        out.loc[dates.isna(), "vacancy_month"] = pd.NA
        return out

    def flag_impossible_dates(self, df: Any, date_columns: list[str]) -> Any:
        """Flag dates outside a valid project-specific range."""
        pd = _require_pandas()
        out = df.copy()
        min_date = pd.Timestamp(DEFAULT_MIN_DATE, tz="UTC")
        max_date = pd.Timestamp(DEFAULT_MAX_DATE, tz="UTC")
        existing_columns = [column for column in date_columns if column in out.columns]

        issue_frames = []
        for column in existing_columns:
            raw_source_column = f"{column}_raw" if f"{column}_raw" in out.columns else column
            raw = out[raw_source_column]
            parsed = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
            raw_text = raw.astype("string").str.strip().str.strip('"').str.strip("'")
            leading_year = raw_text.str.extract(r"^([+-]?\d{3,4})", expand=False)
            leading_year = pd.to_numeric(leading_year, errors="coerce")
            too_early = parsed.notna() & parsed.lt(min_date)
            too_late = parsed.notna() & parsed.gt(max_date)
            parsed_bad_year = parsed.notna() & (
                parsed.dt.year.lt(2000) | parsed.dt.year.gt(max_date.year)
            )
            raw_bad_year = parsed.isna() & (
                leading_year.lt(2000) | leading_year.gt(max_date.year)
            )
            too_early = too_early.fillna(False)
            too_late = too_late.fillna(False)
            bad_year = (parsed_bad_year | raw_bad_year).fillna(False)

            out[f"{column}_is_before_min_date"] = too_early
            out[f"{column}_is_after_max_date"] = too_late
            out[f"{column}_has_impossible_year"] = bad_year
            issue_frames.append(too_early | too_late | bad_year)

        if issue_frames:
            combined = issue_frames[0].copy()
            for mask in issue_frames[1:]:
                combined = combined | mask
            out["has_impossible_date"] = combined
        else:
            out["has_impossible_date"] = False

        return out
