"""Canonical vacancy schema transformations."""

import json
from typing import Any

from .constants import (
    RAW_DATASET_PATHS,
    SOURCE_COMBINED,
    SOURCE_DATASET_LABELS,
    SOURCE_HH_GITHUB,
    SOURCE_HH_KAGGLE,
    SOURCE_TRUDVSEM_LATEST,
    SourceDataset,
    validate_source_dataset,
)


CANONICAL_VACANCY_COLUMNS = [
    "source_dataset",
    "source_name",
    "source_file",
    "source_vacancy_id",
    "source_url",
    "vacancy_date",
    "vacancy_date_raw",
    "vacancy_date_type",
    "snapshot_date",
    "is_open",
    "title_raw",
    "title_normalized",
    "occupation_raw",
    "occupation_group",
    "occupation_code",
    "professional_sphere_raw",
    "professional_sphere_code",
    "employer_name",
    "employer_id",
    "employer_type",
    "industry_raw",
    "industry_group",
    "country",
    "region",
    "region_id",
    "federal_district",
    "city",
    "address_raw",
    "lat",
    "lon",
    "salary_from",
    "salary_to",
    "salary_mid",
    "salary_raw",
    "salary_currency",
    "salary_period",
    "salary_is_missing",
    "salary_bound_type",
    "salary_parse_status",
    "experience_raw",
    "experience_min_years",
    "experience_max_years",
    "experience_group",
    "employment_raw",
    "employment_type",
    "schedule_raw",
    "schedule_type",
    "is_remote",
    "is_shift",
    "is_fly_in_fly_out",
    "description_raw",
    "requirements_raw",
    "responsibilities_raw",
    "skills_raw",
    "skills_normalized",
    "education_raw",
    "education_level",
    "driver_license_required",
    "medical_card_required",
    "accommodation_available",
    "created_at_raw",
    "published_at_raw",
    "modified_at_raw",
    "last_seen_at_raw",
    "source_row_hash",
    "duplicate_group_id",
    "is_duplicate",
    "parse_warnings",
    "quality_flags",
]

REQUIRED_CANONICAL_COLUMNS = [
    "source_dataset",
    "source_vacancy_id",
]


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("CanonicalTransformer requires pandas.") from exc
    return pd


def _empty_canonical_frame(index: Any) -> Any:
    pd = _require_pandas()
    return pd.DataFrame(index=index, columns=CANONICAL_VACANCY_COLUMNS)


def _column_or_na(df: Any, column: str) -> Any:
    pd = _require_pandas()
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index)


def _clean_quoted(value: Any) -> Any:
    pd = _require_pandas()
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    if text.casefold() in {"", "none", "nan", "null"}:
        return pd.NA
    return text


def _clean_quoted_series(series: Any) -> Any:
    return series.map(_clean_quoted)


def _coerce_bool_series(series: Any) -> Any:
    pd = _require_pandas()
    normalized = series.astype("string").str.strip().str.strip('"').str.casefold()
    return normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "да": True,
            "false": False,
            "0": False,
            "no": False,
            "нет": False,
        }
    ).astype("boolean")


def _extract_json_key(value: Any, key: str) -> Any:
    pd = _require_pandas()
    if pd.isna(value):
        return pd.NA
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return pd.NA
    if not isinstance(parsed, dict):
        return pd.NA
    return parsed.get(key, pd.NA)


def _extract_json_series_key(series: Any, key: str) -> Any:
    return series.map(lambda value: _extract_json_key(value, key))


class CanonicalTransformer:
    """Transform each raw source into the canonical vacancy schema."""

    def to_canonical_schema(self, df: Any, source_dataset: str) -> Any:
        """Dispatch a raw dataframe to the matching source transformer."""
        source = validate_source_dataset(source_dataset)
        transformers = {
            SOURCE_TRUDVSEM_LATEST: self.to_canonical_trudvsem,
            SOURCE_HH_GITHUB: self.to_canonical_hh_github,
            SOURCE_HH_KAGGLE: self.to_canonical_hh_kaggle,
            SOURCE_COMBINED: self.to_canonical_combined,
        }
        canonical = transformers[source](df)
        return self.add_source_metadata(
            canonical,
            source,
            str(RAW_DATASET_PATHS.get(source, "")),
        )

    def to_canonical_trudvsem(self, df: Any) -> Any:
        """Transform Trudvsem rows to canonical vacancy rows."""
        out = _empty_canonical_frame(df.index)
        geo = _column_or_na(df, "geo")

        out["source_vacancy_id"] = _column_or_na(df, "id")
        out["source_url"] = _column_or_na(df, "vacancyUrl")
        out["is_open"] = ~_coerce_bool_series(_column_or_na(df, "deleted")).fillna(False)
        out["vacancy_date_raw"] = _column_or_na(df, "creationDate")
        out["vacancy_date_type"] = "created_at"
        out["title_raw"] = _column_or_na(df, "vacancyName")
        out["occupation_raw"] = _column_or_na(df, "typicalPosition")
        out["occupation_code"] = _column_or_na(df, "codeProfession")
        out["professional_sphere_raw"] = _column_or_na(df, "professionalSphereName")
        out["professional_sphere_code"] = _column_or_na(df, "codeProfessionalSphere")
        out["employer_name"] = _column_or_na(df, "fullCompanyName")
        out["employer_id"] = _extract_json_series_key(_column_or_na(df, "company"), "companyCode")
        out["employer_type"] = _column_or_na(df, "companyBusinessSize")
        out["industry_raw"] = _column_or_na(df, "industryBranchName")
        out["country"] = "Россия"
        out["region"] = _column_or_na(df, "regionName")
        out["region_id"] = _column_or_na(df, "stateRegionCode")
        out["federal_district"] = _column_or_na(df, "federalDistrictCode")
        out["address_raw"] = _column_or_na(df, "vacancyAddress")
        out["lat"] = _extract_json_series_key(geo, "latitude")
        out["lon"] = _extract_json_series_key(geo, "longitude")
        out["salary_from"] = _column_or_na(df, "salaryMin")
        out["salary_to"] = _column_or_na(df, "salaryMax")
        out["salary_raw"] = _column_or_na(df, "salary")
        out["experience_raw"] = _column_or_na(df, "experienceRequirements")
        out["employment_raw"] = _column_or_na(df, "busyType")
        out["schedule_raw"] = _column_or_na(df, "scheduleType")
        out["description_raw"] = _column_or_na(df, "positionRequirements")
        out["requirements_raw"] = _column_or_na(df, "qualifications")
        out["responsibilities_raw"] = _column_or_na(df, "responsibilities")
        out["skills_raw"] = _column_or_na(df, "skills")
        out["education_raw"] = _column_or_na(df, "educationRequirements")
        out["driver_license_required"] = _column_or_na(df, "requiredDriveLicense")
        out["medical_card_required"] = _column_or_na(df, "needMedcard")
        out["accommodation_available"] = _column_or_na(df, "accommodationCapability")
        out["created_at_raw"] = _column_or_na(df, "creationDate")
        out["published_at_raw"] = _column_or_na(df, "datePublished")
        out["modified_at_raw"] = _column_or_na(df, "dateModify")
        out["last_seen_at_raw"] = _column_or_na(df, "changeTime")
        return out

    def to_canonical_hh_github(self, df: Any) -> Any:
        """Transform HH GitHub rows to canonical vacancy rows."""
        out = _empty_canonical_frame(df.index)
        out["source_vacancy_id"] = _clean_quoted_series(_column_or_na(df, "id"))
        out["vacancy_date_raw"] = _clean_quoted_series(_column_or_na(df, "published_at"))
        out["vacancy_date_type"] = "published_at"
        out["title_raw"] = _clean_quoted_series(_column_or_na(df, "name"))
        out["occupation_raw"] = _clean_quoted_series(_column_or_na(df, "professional_role"))
        out["employer_name"] = _clean_quoted_series(_column_or_na(df, "employer_name"))
        out["city"] = _clean_quoted_series(_column_or_na(df, "city"))
        out["salary_from"] = _clean_quoted_series(_column_or_na(df, "salary_bottom"))
        out["salary_to"] = _clean_quoted_series(_column_or_na(df, "salary_top"))
        out["salary_currency"] = _clean_quoted_series(_column_or_na(df, "currency"))
        out["experience_raw"] = _clean_quoted_series(_column_or_na(df, "experience"))
        out["schedule_raw"] = _clean_quoted_series(_column_or_na(df, "schedule"))
        out["skills_raw"] = _clean_quoted_series(_column_or_na(df, "key_skills"))
        out["published_at_raw"] = out["vacancy_date_raw"]
        return out

    def to_canonical_hh_kaggle(self, df: Any) -> Any:
        """Transform HH Kaggle rows to canonical vacancy rows."""
        out = _empty_canonical_frame(df.index)
        out["source_vacancy_id"] = _column_or_na(df, "id")
        out["is_open"] = _column_or_na(df, "type").astype("string").str.casefold().ne("close")
        out["vacancy_date_raw"] = _column_or_na(df, "date_of_post")
        out["vacancy_date_type"] = "published_at"
        out["title_raw"] = _column_or_na(df, "title")
        out["employer_name"] = _column_or_na(df, "company")
        out["city"] = _column_or_na(df, "location")
        out["address_raw"] = _column_or_na(df, "location")
        out["salary_raw"] = _column_or_na(df, "salary")
        out["experience_raw"] = _column_or_na(df, "experience")
        out["employment_raw"] = _column_or_na(df, "job_type")
        out["schedule_raw"] = _column_or_na(df, "job_type")
        out["description_raw"] = _column_or_na(df, "description")
        out["skills_raw"] = _column_or_na(df, "key_skills")
        out["published_at_raw"] = _column_or_na(df, "date_of_post")
        return out

    def to_canonical_combined(self, df: Any) -> Any:
        """Transform HH/Trudvsem/Mendeley combined rows to canonical vacancy rows."""
        out = _empty_canonical_frame(df.index)
        out["source_vacancy_id"] = _column_or_na(df, "id")
        out["source_url"] = _column_or_na(df, "link")
        out["is_open"] = _coerce_bool_series(_column_or_na(df, "is_open"))
        out["vacancy_date_raw"] = _column_or_na(df, "last_found_at")
        out["vacancy_date_type"] = "last_seen_at"
        out["title_raw"] = _column_or_na(df, "name")
        out["occupation_raw"] = _column_or_na(df, "role_name")
        out["occupation_code"] = _column_or_na(df, "role_id")
        out["employer_name"] = _column_or_na(df, "employer_name")
        out["employer_id"] = _column_or_na(df, "employer_id")
        out["employer_type"] = _column_or_na(df, "employer_type")
        out["industry_raw"] = _column_or_na(df, "employer_industry_name")
        out["country"] = _column_or_na(df, "country_name")
        out["region"] = _column_or_na(df, "region_name")
        out["region_id"] = _column_or_na(df, "region_id")
        out["salary_from"] = _column_or_na(df, "salary_from")
        out["salary_to"] = _column_or_na(df, "salary_to")
        out["salary_raw"] = _column_or_na(df, "salary")
        out["experience_raw"] = _column_or_na(df, "experience")
        out["employment_raw"] = _column_or_na(df, "employment")
        out["schedule_raw"] = _column_or_na(df, "schedule")
        out["description_raw"] = _column_or_na(df, "description")
        out["requirements_raw"] = _column_or_na(df, "requirements")
        out["skills_raw"] = _column_or_na(df, "raw_skills")
        out["accommodation_available"] = _column_or_na(df, "accomodation")
        out["last_seen_at_raw"] = _column_or_na(df, "last_found_at")
        return out

    def add_source_metadata(self, df: Any, source_dataset: str, source_file: str | None = None) -> Any:
        """Attach stable source metadata columns."""
        source = validate_source_dataset(source_dataset)
        out = df.copy()
        out["source_dataset"] = source
        out["source_name"] = SOURCE_DATASET_LABELS[source]
        out["source_file"] = source_file
        return out.reindex(columns=CANONICAL_VACANCY_COLUMNS)

    def validate_canonical_schema(self, df: Any) -> Any:
        """Validate required canonical columns and basic types."""
        missing_columns = [
            column for column in CANONICAL_VACANCY_COLUMNS if column not in df.columns
        ]
        extra_columns = [
            column for column in df.columns if column not in CANONICAL_VACANCY_COLUMNS
        ]
        required_missing_values = {
            column: int(df[column].isna().sum())
            for column in REQUIRED_CANONICAL_COLUMNS
            if column in df.columns
        }
        source_values = (
            sorted(df["source_dataset"].dropna().unique().tolist())
            if "source_dataset" in df.columns
            else []
        )
        invalid_sources = [
            source
            for source in source_values
            if source
            not in {
                SOURCE_TRUDVSEM_LATEST,
                SOURCE_HH_GITHUB,
                SOURCE_HH_KAGGLE,
                SOURCE_COMBINED,
            }
        ]

        errors = []
        if missing_columns:
            errors.append(f"Missing canonical columns: {missing_columns}")
        if invalid_sources:
            errors.append(f"Unsupported source_dataset values: {invalid_sources}")

        return {
            "is_valid": not errors,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "required_missing_values": required_missing_values,
            "invalid_sources": invalid_sources,
            "errors": errors,
        }
