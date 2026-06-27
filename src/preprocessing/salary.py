"""Salary parsing and normalization utilities."""

import re
from typing import Any

from .constants import (
    SOURCE_COMBINED,
    SOURCE_HH_GITHUB,
    SOURCE_HH_KAGGLE,
    SOURCE_TRUDVSEM_LATEST,
    validate_source_dataset,
)


MISSING_SALARY_STRINGS = {
    "",
    "none",
    "nan",
    "null",
    "з/п не указана",
    "зарплата не указана",
    "не указана",
    "не указано",
}

SALARY_EXTREME_THRESHOLD = 1_000_000


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("SalaryProcessor requires pandas.") from exc
    return pd


def _clean_text(value: Any) -> Any:
    pd = _require_pandas()
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().strip('"').strip("'")
    if text.casefold() in MISSING_SALARY_STRINGS:
        return pd.NA
    return text


def _clean_series(series: Any) -> Any:
    return series.map(_clean_text)


def _to_numeric_series(series: Any) -> Any:
    pd = _require_pandas()
    cleaned = _clean_series(series).astype("string")
    cleaned = cleaned.str.replace(r"[^\d,.\-]", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_hh_kaggle_salary(value: Any) -> dict[str, Any]:
    pd = _require_pandas()
    text = _clean_text(value)
    if pd.isna(text):
        return {
            "salary_from": pd.NA,
            "salary_to": pd.NA,
            "salary_currency": pd.NA,
            "salary_parse_status": "missing",
        }

    raw = str(text)
    folded = raw.casefold()
    if folded in MISSING_SALARY_STRINGS:
        return {
            "salary_from": pd.NA,
            "salary_to": pd.NA,
            "salary_currency": pd.NA,
            "salary_parse_status": "missing",
        }

    numbers = [
        float(item.replace(" ", "").replace(",", "."))
        for item in re.findall(r"\d[\d\s]*(?:[,.]\d+)?", raw)
    ]
    currency_match = re.search(r"\b([A-Z]{3})\b", raw)
    currency = currency_match.group(1) if currency_match else pd.NA

    salary_from = pd.NA
    salary_to = pd.NA
    if "от" in folded and "до" in folded and len(numbers) >= 2:
        salary_from = numbers[0]
        salary_to = numbers[1]
    elif "от" in folded and numbers:
        salary_from = numbers[0]
    elif "до" in folded and numbers:
        salary_to = numbers[0]
    elif len(numbers) >= 2:
        salary_from = numbers[0]
        salary_to = numbers[1]
    elif len(numbers) == 1:
        salary_from = numbers[0]

    status = "parsed" if not pd.isna(salary_from) or not pd.isna(salary_to) else "unparsed"
    return {
        "salary_from": salary_from,
        "salary_to": salary_to,
        "salary_currency": currency,
        "salary_parse_status": status,
    }


class SalaryProcessor:
    """Parse salaries into canonical bounds and quality flags."""

    def parse_salary(self, df: Any, source_dataset: str) -> Any:
        """Dispatch salary parsing by source dataset."""
        source = validate_source_dataset(source_dataset)
        out = df.copy()

        if source == SOURCE_TRUDVSEM_LATEST:
            out["salary_from"] = _to_numeric_series(out["salary_from"])
            out["salary_to"] = _to_numeric_series(out["salary_to"])
            out["salary_raw"] = _clean_series(out["salary_raw"])
            out["salary_currency"] = "RUR"
            out["salary_parse_status"] = "parsed"
        elif source == SOURCE_HH_GITHUB:
            out["salary_from"] = _to_numeric_series(out["salary_from"])
            out["salary_to"] = _to_numeric_series(out["salary_to"])
            out["salary_currency"] = _clean_series(out["salary_currency"])
            out["salary_parse_status"] = "parsed"
        elif source == SOURCE_HH_KAGGLE:
            parsed = out["salary_raw"].map(_parse_hh_kaggle_salary)
            parsed_df = _require_pandas().DataFrame(parsed.tolist(), index=out.index)
            for column in parsed_df.columns:
                out[column] = parsed_df[column]
        elif source == SOURCE_COMBINED:
            out["salary_from"] = _to_numeric_series(out["salary_from"])
            out["salary_to"] = _to_numeric_series(out["salary_to"])
            out["salary_raw"] = _clean_series(out["salary_raw"])
            out["salary_parse_status"] = "parsed"

        out = self.normalize_salary_bounds(out)
        out = self.compute_salary_mid(out)
        out["salary_bound_type"] = out.apply(self.detect_salary_bound_type, axis=1)
        out["salary_is_missing"] = out["salary_bound_type"].eq("missing")
        out = self.flag_salary_quality(out)
        return out

    def normalize_salary_bounds(
        self,
        df: Any,
        from_column: str = "salary_from",
        to_column: str = "salary_to",
    ) -> Any:
        """Normalize salary bounds and handle zero/missing values."""
        pd = _require_pandas()
        out = df.copy()
        for column in [from_column, to_column]:
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce")
                out.loc[out[column].le(0), column] = pd.NA
        return out

    def compute_salary_mid(
        self,
        df: Any,
        from_column: str = "salary_from",
        to_column: str = "salary_to",
        output_column: str = "salary_mid",
    ) -> Any:
        """Compute midpoint salary where both bounds are valid."""
        out = df.copy()
        lower = out[from_column]
        upper = out[to_column]
        valid_range = lower.notna() & upper.notna() & lower.gt(0) & upper.gt(0) & lower.le(upper)
        out[output_column] = _require_pandas().NA
        out.loc[valid_range, output_column] = (lower[valid_range] + upper[valid_range]) / 2
        return out

    def flag_salary_quality(self, df: Any) -> Any:
        """Flag missing, zero, inconsistent, and extreme salary records."""
        out = df.copy()
        lower = out["salary_from"]
        upper = out["salary_to"]
        currency = out["salary_currency"] if "salary_currency" in out.columns else None

        out["salary_has_missing_bounds"] = lower.isna() & upper.isna()
        out["salary_has_zero_bound"] = lower.eq(0).fillna(False) | upper.eq(0).fillna(False)
        out["salary_has_inverted_bounds"] = lower.notna() & upper.notna() & lower.gt(upper)
        out["salary_has_extreme_value"] = (
            lower.gt(SALARY_EXTREME_THRESHOLD).fillna(False)
            | upper.gt(SALARY_EXTREME_THRESHOLD).fillna(False)
        )
        if currency is not None:
            out["salary_is_non_rur"] = currency.notna() & currency.astype("string").str.upper().ne("RUR")
        else:
            out["salary_is_non_rur"] = False
        return out

    def detect_salary_bound_type(self, row: Any) -> str:
        """Return range/lower_only/upper_only/missing for a salary row."""
        pd = _require_pandas()
        lower = row.get("salary_from", pd.NA)
        upper = row.get("salary_to", pd.NA)

        has_lower = not pd.isna(lower)
        has_upper = not pd.isna(upper)
        if has_lower and has_upper:
            if lower <= 0 or upper <= 0 or lower > upper:
                return "invalid"
            return "range"
        if has_lower:
            return "lower_only" if lower > 0 else "invalid"
        if has_upper:
            return "upper_only" if upper > 0 else "invalid"
        return "missing"
