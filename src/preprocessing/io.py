"""Dataset reading and audit helpers."""

import csv
import sqlite3
from pathlib import Path
from typing import Any

from .constants import (
    RAW_DATASET_PATHS,
    SOURCE_COMBINED,
    SOURCE_HH_GITHUB,
    SOURCE_HH_KAGGLE,
    SOURCE_TRUDVSEM_LATEST,
    SourceDataset,
    validate_source_dataset,
)


HH_GITHUB_EXPECTED_COLUMNS = {
    "id",
    "name",
    "city",
    "salary_bottom",
    "salary_top",
    "currency",
    "published_at",
    "employer_name",
    "key_skills",
    "schedule",
    "professional_role",
    "experience",
}

MISSING_VALUE_PLACEHOLDERS = {
    "",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "нет",
    "не задано",
    "не указано",
    "з/п не указана",
    "зарплата не указана",
}


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("DatasetReader requires pandas to read raw datasets.") from exc
    return pd


def _resolve_existing_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {resolved}")
    return resolved


def _clean_string_values(series: Any) -> Any:
    return series.astype("string").str.strip().str.strip('"').str.strip("'")


def _normalize_for_placeholder(series: Any) -> Any:
    return _clean_string_values(series).str.casefold()


def _looks_dayfirst(series: Any) -> bool:
    sample = _clean_string_values(series).dropna().head(100)
    if sample.empty:
        return False
    return bool(sample.str.match(r"^\d{1,2}\.\d{1,2}\.\d{2,4}").any())


def _existing_columns(df: Any, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _csv_read_options(**kwargs: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "dtype": str,
        "encoding": "utf-8-sig",
        "keep_default_na": True,
    }
    options.update(kwargs)
    return options


def _merge_trudvsem_surplus_fields(fields: list[str]) -> list[str]:
    header = TRUDVSEM_EXPECTED_COLUMNS
    if len(fields) <= len(header):
        return fields

    merge_column = "responsibilities"
    merge_index = header.index(merge_column)
    surplus = len(fields) - len(header)
    merged_value = "|".join(fields[merge_index : merge_index + surplus + 1])
    return fields[:merge_index] + [merged_value] + fields[merge_index + surplus + 1 :]


TRUDVSEM_EXPECTED_COLUMNS = [
    "id",
    "stateRegionCode",
    "vacancyName",
    "codeProfession",
    "codeProfessionalSphere",
    "professionalSphereName",
    "vacancyAddress",
    "vacancyAddressHouse",
    "vacancyAddressAdditionalInfo",
    "geo",
    "salary",
    "socialProtecteds",
    "languageKnowledge",
    "busyType",
    "educationRequirements",
    "hardSkills",
    "softSkills",
    "skills",
    "typicalPosition",
    "experienceRequirements",
    "scheduleType",
    "premium",
    "otherVacancyBenefit",
    "careerPerspective",
    "codeExternalSystem",
    "idPriorityCategory",
    "needMedcard",
    "sourceType",
    "requiredСertificates",
    "requiredDriveLicense",
    "retrainingCapability",
    "retrainingСondition",
    "retrainingGrantValue",
    "transportCompensation",
    "changeTime",
    "contactPerson",
    "contactSource",
    "company",
    "fullCompanyName",
    "oknpoCode",
    "oksoCode",
    "companyBusinessSize",
    "dateModify",
    "workPlaces",
    "isUzbekistanRecruitment",
    "federalDistrictCode",
    "industryBranchName",
    "datePublished",
    "accommodationCapability",
    "accommodationType",
    "foreignWorkersCapability",
    "metroIds",
    "isQuoted",
    "creationDate",
    "isMobilityProgram",
    "isModerated",
    "deleted",
    "visibility",
    "regionName",
    "status",
    "vacancyUrl",
    "positionRequirements",
    "contactList",
    "additionalRequirements",
    "salaryMin",
    "salaryMax",
    "qualifications",
    "responsibilities",
    "addressCode",
    "addressOffice",
    "hireDate",
    "workPlace",
    "medicalCertificate",
    "scheduleTypeComment",
    "benefitDetails",
    "trainingDays",
    "shift",
    "medicalDocument",
    "benefit",
    "conditions",
]


class DatasetReader:
    """Read raw vacancy datasets with source-specific parser settings."""

    def read_dataset(self, path: str | Path, source_dataset: str, **kwargs: Any) -> Any:
        """Dispatch reading by source dataset name."""
        source = validate_source_dataset(source_dataset)
        readers = {
            SOURCE_TRUDVSEM_LATEST: self.read_trudvsem_csv,
            SOURCE_HH_GITHUB: self.read_hh_github_sqlite,
            SOURCE_HH_KAGGLE: self.read_hh_kaggle_csv,
            SOURCE_COMBINED: self.read_combined_csv,
        }
        return readers[source](path, **kwargs)

    def read_trudvsem_csv(self, path: str | Path, **kwargs: Any) -> Any:
        """Read the pipe-delimited Trudvsem CSV."""
        pd = _require_pandas()
        csv_path = _resolve_existing_path(path)
        options = _csv_read_options(**kwargs)
        if "engine" not in options and "on_bad_lines" not in options:
            options["engine"] = "python"
            options["on_bad_lines"] = _merge_trudvsem_surplus_fields
        options.setdefault("quoting", csv.QUOTE_NONE)
        return pd.read_csv(csv_path, sep="|", **options)

    def read_hh_kaggle_csv(self, path: str | Path, **kwargs: Any) -> Any:
        """Read the semicolon-delimited HH Kaggle CSV."""
        pd = _require_pandas()
        csv_path = _resolve_existing_path(path)
        return pd.read_csv(csv_path, sep=";", **_csv_read_options(**kwargs))

    def read_hh_github_sqlite(self, path: str | Path, table: str = "vacancies", **kwargs: Any) -> Any:
        """Read HH GitHub vacancies from SQLite."""
        pd = _require_pandas()
        sqlite_path = _resolve_existing_path(path)
        limit = kwargs.pop("limit", None)

        query = f'SELECT * FROM "{table}"'
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError("limit must be a non-negative integer.")
            query = f"{query} LIMIT {limit}"

        with sqlite3.connect(sqlite_path) as connection:
            available_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if table not in available_tables:
                raise ValueError(
                    f"SQLite table {table!r} not found in {sqlite_path}. "
                    f"Available tables: {sorted(available_tables)}"
                )

            table_columns = {
                row[1]
                for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            }
            missing_columns = sorted(HH_GITHUB_EXPECTED_COLUMNS - table_columns)
            if missing_columns:
                raise ValueError(
                    f"SQLite table {table!r} is missing expected columns: "
                    f"{missing_columns}"
                )

            df = pd.read_sql_query(query, connection, **kwargs)

        if "id" in df.columns:
            df["id"] = df["id"].astype("string")
        return df

    def read_combined_csv(self, path: str | Path, **kwargs: Any) -> Any:
        """Read the HH/Trudvsem/Mendeley combined CSV."""
        pd = _require_pandas()
        csv_path = _resolve_existing_path(path)
        return pd.read_csv(csv_path, sep=";", **_csv_read_options(**kwargs))


class DataProfiler:
    """Small reusable profiling helpers for raw and canonical dataframes."""

    def profile_dataframe(self, df: Any) -> dict[str, Any]:
        """Return row count, columns, dtypes, missingness, dates, and salary summaries."""
        pd = _require_pandas()
        date_columns = self._infer_date_columns(df)
        salary_columns = self._infer_salary_columns(df)

        missing = self.summarize_missing_values(df)
        missing_rates = (
            missing.set_index("column")["total_missing_rate"].to_dict()
            if not missing.empty
            else {}
        )

        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "missing_rates": missing_rates,
            "missing_values": missing,
            "date_coverage": self.summarize_date_coverage(df, date_columns),
            "salary_columns": self.summarize_salary_columns(df, salary_columns),
            "duplicate_summary": self.summarize_duplicates(df, self._infer_id_columns(df)),
        }

    def summarize_missing_values(self, df: Any) -> Any:
        """Summarize null and placeholder-missing values."""
        pd = _require_pandas()
        rows: list[dict[str, Any]] = []
        row_count = len(df)

        for column in df.columns:
            series = df[column]
            nan_count = int(series.isna().sum())

            if pd.api.types.is_string_dtype(series) or series.dtype == "object":
                normalized = _normalize_for_placeholder(series)
                non_null = series.notna()
                empty_count = int((non_null & normalized.eq("")).sum())
                placeholder_mask = non_null & normalized.isin(MISSING_VALUE_PLACEHOLDERS - {""})
                placeholder_count = int(placeholder_mask.sum())
                placeholder_examples = sorted(
                    normalized[placeholder_mask].dropna().unique().tolist()
                )[:10]
            else:
                empty_count = 0
                placeholder_count = 0
                placeholder_examples = []

            total_missing = nan_count + empty_count + placeholder_count
            rows.append(
                {
                    "column": column,
                    "nan_count": nan_count,
                    "empty_string_count": empty_count,
                    "placeholder_count": placeholder_count,
                    "total_missing_count": total_missing,
                    "total_missing_rate": total_missing / row_count if row_count else 0.0,
                    "placeholder_examples": placeholder_examples,
                }
            )

        return pd.DataFrame(rows)

    def summarize_duplicates(self, df: Any, id_columns: list[str]) -> Any:
        """Summarize full-row and ID-like duplicates."""
        pd = _require_pandas()
        row_count = len(df)
        rows = [
            {
                "scope": "full_row",
                "columns": list(df.columns),
                "duplicate_rows": int(df.duplicated(keep=False).sum()),
                "duplicate_groups": int(df[df.duplicated(keep=False)].drop_duplicates().shape[0]),
                "non_empty_rows": row_count,
                "duplicate_rate": (
                    int(df.duplicated(keep=False).sum()) / row_count if row_count else 0.0
                ),
            }
        ]

        for column in _existing_columns(df, id_columns):
            series = df[column]
            non_empty = series.notna()
            if pd.api.types.is_string_dtype(series) or series.dtype == "object":
                normalized = _normalize_for_placeholder(series)
                non_empty &= ~normalized.isin(MISSING_VALUE_PLACEHOLDERS)

            duplicate_mask = non_empty & series.duplicated(keep=False)
            duplicate_values = series[duplicate_mask]
            rows.append(
                {
                    "scope": column,
                    "columns": [column],
                    "duplicate_rows": int(duplicate_mask.sum()),
                    "duplicate_groups": int(duplicate_values.nunique(dropna=True)),
                    "non_empty_rows": int(non_empty.sum()),
                    "duplicate_rate": (
                        int(duplicate_mask.sum()) / int(non_empty.sum())
                        if int(non_empty.sum())
                        else 0.0
                    ),
                }
            )

        return pd.DataFrame(rows)

    def summarize_date_coverage(self, df: Any, date_columns: list[str]) -> Any:
        """Summarize min/max and yearly coverage for date columns."""
        pd = _require_pandas()
        rows: list[dict[str, Any]] = []

        for column in _existing_columns(df, date_columns):
            raw = df[column]
            values = _clean_string_values(raw) if raw.dtype == "object" else raw
            parsed = pd.to_datetime(
                values,
                errors="coerce",
                utc=True,
                dayfirst=_looks_dayfirst(raw) if raw.dtype == "object" else False,
            )
            non_empty_count = int(raw.notna().sum())
            parsed_count = int(parsed.notna().sum())
            years = parsed.dropna().dt.year.value_counts().sort_index()

            rows.append(
                {
                    "column": column,
                    "non_empty_count": non_empty_count,
                    "parsed_count": parsed_count,
                    "parse_success_rate": (
                        parsed_count / non_empty_count if non_empty_count else 0.0
                    ),
                    "min": parsed.min().isoformat() if parsed_count else None,
                    "max": parsed.max().isoformat() if parsed_count else None,
                    "year_counts": {
                        int(year): int(count) for year, count in years.items()
                    },
                }
            )

        return pd.DataFrame(rows)

    def summarize_salary_columns(self, df: Any, salary_columns: list[str]) -> Any:
        """Summarize salary completeness, zeros, ranges, and outliers."""
        pd = _require_pandas()
        rows: list[dict[str, Any]] = []

        for column in _existing_columns(df, salary_columns):
            raw = df[column]
            if pd.api.types.is_string_dtype(raw) or raw.dtype == "object":
                cleaned = _clean_string_values(raw).str.replace(" ", "", regex=False)
                numeric = pd.to_numeric(cleaned, errors="coerce")
            else:
                numeric = pd.to_numeric(raw, errors="coerce")

            non_empty_count = int(raw.notna().sum())
            numeric_count = int(numeric.notna().sum())
            zero_count = int(numeric.eq(0).sum())
            negative_count = int(numeric.lt(0).sum())
            positive = numeric[numeric > 0]
            q1 = positive.quantile(0.25) if not positive.empty else None
            q3 = positive.quantile(0.75) if not positive.empty else None
            iqr = (q3 - q1) if q1 is not None and q3 is not None else None
            iqr_threshold = (q3 + 3 * iqr) if iqr is not None else None
            extreme_threshold = max(1_000_000, iqr_threshold or 0)

            rows.append(
                {
                    "column": column,
                    "non_empty_count": non_empty_count,
                    "numeric_count": numeric_count,
                    "numeric_rate": numeric_count / non_empty_count if non_empty_count else 0.0,
                    "zero_count": zero_count,
                    "negative_count": negative_count,
                    "min": float(numeric.min()) if numeric_count else None,
                    "max": float(numeric.max()) if numeric_count else None,
                    "median": float(numeric.median()) if numeric_count else None,
                    "p25": float(positive.quantile(0.25)) if not positive.empty else None,
                    "p75": float(positive.quantile(0.75)) if not positive.empty else None,
                    "potential_outlier_count": int((numeric > extreme_threshold).sum())
                    if numeric_count
                    else 0,
                    "extreme_threshold": float(extreme_threshold) if numeric_count else None,
                }
            )

        return pd.DataFrame(rows)

    def _infer_date_columns(self, df: Any) -> list[str]:
        pd = _require_pandas()
        columns: list[str] = []
        for column in df.columns:
            name = str(column).casefold()
            if (
                "date" in name
                or "time" in name
                or "дата" in name
                or pd.api.types.is_datetime64_any_dtype(df[column])
            ):
                columns.append(column)
        return columns

    def _infer_salary_columns(self, df: Any) -> list[str]:
        salary_tokens = ("salary", "зарп", "з/п", "оклад")
        return [
            column
            for column in df.columns
            if any(token in str(column).casefold() for token in salary_tokens)
        ]

    def _infer_id_columns(self, df: Any) -> list[str]:
        candidates = [
            "id",
            "source_id",
            "vacancy_id",
            "vacancyUrl",
            "link",
            "Unnamed: 0",
            "№",
        ]
        return _existing_columns(df, candidates)


def profile_raw_datasets(sample_rows: int | None = None) -> dict[SourceDataset, dict[str, Any]]:
    """Read and profile all configured raw datasets one by one."""
    reader = DatasetReader()
    profiler = DataProfiler()
    reports: dict[SourceDataset, dict[str, Any]] = {}

    for source_dataset, path in RAW_DATASET_PATHS.items():
        kwargs: dict[str, Any] = {}
        if sample_rows is not None:
            if source_dataset == SOURCE_HH_GITHUB:
                kwargs["limit"] = sample_rows
            else:
                kwargs["nrows"] = sample_rows
        df = reader.read_dataset(path, source_dataset, **kwargs)
        reports[source_dataset] = profiler.profile_dataframe(df)

    return reports
