"""Quality checks and deduplication helpers."""

import hashlib
import json
import re
from typing import Any

from .canonical import CANONICAL_VACANCY_COLUMNS
from .constants import validate_source_dataset


DEFAULT_HASH_COLUMNS = [
    "source_dataset",
    "source_vacancy_id",
    "source_url",
    "title_raw",
    "employer_name",
    "region",
    "city",
    "vacancy_date_raw",
    "salary_from",
    "salary_to",
]


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("QualityChecker requires pandas.") from exc
    return pd


def _normalize_value(value: Any) -> str:
    pd = _require_pandas()
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def _row_hash(values: list[Any]) -> str:
    normalized = [_normalize_value(value) for value in values]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_flag(flags: list[str], condition: bool, flag: str) -> list[str]:
    if condition:
        flags.append(flag)
    return flags


class QualityChecker:
    """Create source keys, duplicate flags, and merge-readiness checks."""

    def make_source_key(self, df: Any, source_dataset: str) -> Any:
        """Create a stable source-level vacancy key."""
        pd = _require_pandas()
        source = validate_source_dataset(source_dataset)
        out = df.copy()

        if "source_dataset" not in out.columns:
            out["source_dataset"] = source

        source_id = (
            out["source_vacancy_id"].astype("string")
            if "source_vacancy_id" in out.columns
            else pd.Series(pd.NA, index=out.index, dtype="string")
        )
        source_id = source_id.str.strip()
        has_id = source_id.notna() & source_id.ne("")

        fallback_columns = [
            column
            for column in ["source_url", "title_raw", "employer_name", "region", "city", "vacancy_date_raw"]
            if column in out.columns
        ]
        fallback_hashes = out[fallback_columns].apply(
            lambda row: _row_hash(row.tolist()),
            axis=1,
        ) if fallback_columns else pd.Series([_row_hash([idx]) for idx in out.index], index=out.index)

        out["source_key"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out.loc[has_id, "source_key"] = source + ":" + source_id[has_id]
        out.loc[~has_id, "source_key"] = source + ":fallback:" + fallback_hashes[~has_id]
        return out

    def make_source_row_hash(self, df: Any, columns: list[str] | None = None) -> Any:
        """Create a hash for duplicate row detection."""
        out = df.copy()
        hash_columns = columns or DEFAULT_HASH_COLUMNS
        existing_columns = [column for column in hash_columns if column in out.columns]
        if not existing_columns:
            raise ValueError("No hash columns are present in dataframe.")

        out["source_row_hash"] = out[existing_columns].apply(
            lambda row: _row_hash(row.tolist()),
            axis=1,
        )
        return out

    def deduplicate_within_source(self, df: Any, source_dataset: str) -> Any:
        """Deduplicate rows within one source dataset."""
        out = self.make_source_key(df, source_dataset)
        out = self.make_source_row_hash(out)

        duplicate_key_mask = out["source_key"].duplicated(keep=False)
        duplicate_hash_mask = out["source_row_hash"].duplicated(keep=False)
        duplicate_mask = duplicate_key_mask | duplicate_hash_mask

        key_group = out.groupby("source_key", dropna=False).ngroup()
        hash_group = out.groupby("source_row_hash", dropna=False).ngroup()

        out["is_duplicate"] = duplicate_mask
        out["duplicate_group_id"] = _require_pandas().NA
        out.loc[duplicate_key_mask, "duplicate_group_id"] = (
            "key:" + key_group[duplicate_key_mask].astype("string")
        )
        hash_only = duplicate_hash_mask & ~duplicate_key_mask
        out.loc[hash_only, "duplicate_group_id"] = (
            "hash:" + hash_group[hash_only].astype("string")
        )
        return out

    def add_quality_flags(self, df: Any) -> Any:
        """Attach parse and quality flags used before modeling."""
        pd = _require_pandas()
        out = df.copy()
        flags: list[list[str]] = []

        for _, row in out.iterrows():
            row_flags: list[str] = []
            _append_flag(
                row_flags,
                bool(row.get("has_impossible_date", False)),
                "impossible_date",
            )
            _append_flag(
                row_flags,
                bool(row.get("vacancy_date_parse_failed", False)),
                "date_parse_failed",
            )
            _append_flag(
                row_flags,
                bool(row.get("salary_has_missing_bounds", False)),
                "salary_missing",
            )
            _append_flag(
                row_flags,
                bool(row.get("salary_has_zero_bound", False)),
                "salary_zero_bound",
            )
            _append_flag(
                row_flags,
                bool(row.get("salary_has_inverted_bounds", False)),
                "salary_inverted_bounds",
            )
            _append_flag(
                row_flags,
                bool(row.get("salary_has_extreme_value", False)),
                "salary_extreme",
            )
            _append_flag(
                row_flags,
                bool(row.get("salary_is_non_rur", False)),
                "salary_non_rur",
            )

            source_id = row.get("source_vacancy_id", pd.NA)
            title = row.get("title_raw", pd.NA)
            region = row.get("region", pd.NA)
            city = row.get("city", pd.NA)

            _append_flag(row_flags, pd.isna(source_id) or str(source_id).strip() == "", "missing_source_id")
            _append_flag(row_flags, pd.isna(title) or str(title).strip() == "", "missing_title")
            _append_flag(row_flags, not pd.isna(title) and len(str(title).strip()) < 3, "short_title")
            _append_flag(
                row_flags,
                (pd.isna(region) or str(region).strip() == "")
                and (pd.isna(city) or str(city).strip() == ""),
                "missing_geography",
            )
            flags.append(row_flags)

        out["quality_flags"] = flags
        out["quality_flag_count"] = [len(item) for item in flags]
        return out

    def check_schema_compatibility(self, dfs: dict[str, Any]) -> Any:
        """Check whether canonical datasets are merge-compatible."""
        reference_columns = list(CANONICAL_VACANCY_COLUMNS)
        extra_sets = {
            name: {column for column in df.columns if column not in reference_columns}
            for name, df in dfs.items()
        }
        reference_extra_columns = next(iter(extra_sets.values()), set())
        reports: dict[str, dict[str, Any]] = {}
        all_errors: list[str] = []

        for name, df in dfs.items():
            columns = list(df.columns)
            missing_columns = [column for column in reference_columns if column not in columns]
            extra_columns = [column for column in columns if column not in reference_columns]
            extra_column_mismatch = sorted(extra_sets[name] ^ reference_extra_columns)
            dtype_report = {column: str(dtype) for column, dtype in df.dtypes.items()}

            errors = []
            if missing_columns:
                errors.append(f"missing columns: {missing_columns}")
            if extra_column_mismatch:
                errors.append(f"extra column mismatch: {extra_column_mismatch}")

            reports[name] = {
                "row_count": int(len(df)),
                "column_count": int(len(columns)),
                "missing_columns": missing_columns,
                "extra_columns": extra_columns,
                "extra_column_mismatch": extra_column_mismatch,
                "dtypes": dtype_report,
                "is_compatible": not errors,
                "errors": errors,
            }
            all_errors.extend([f"{name}: {error}" for error in errors])

        return {
            "is_compatible": not all_errors,
            "reference_columns": reference_columns,
            "reference_extra_columns": sorted(reference_extra_columns),
            "datasets": reports,
            "errors": all_errors,
        }
