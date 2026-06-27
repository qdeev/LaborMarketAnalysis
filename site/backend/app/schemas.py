"""Response models for the public analytics API."""

from pydantic import BaseModel


class FilterOptionsResponse(BaseModel):
    granularity: str
    source_datasets: list[str]
    regions: list[str]
    occupation_groups: list[str]
    periods: list[str]


class OverviewResponse(BaseModel):
    granularity: str
    period: str
    segment_count: int
    median_salary_mid: float | None
    vacancy_count: int
    salary_coverage: float | None
    data_quality_score: float | None


class QuarterlyTrendPoint(BaseModel):
    period: str
    median_salary_mid: float | None
    vacancy_count: int
    salary_coverage: float | None


class SegmentRow(BaseModel):
    source_dataset: str
    region: str
    occupation_group: str
    period: str
    median_salary_mid: float | None
    vacancy_count: int
    salary_count: int


class SegmentHistoryPoint(BaseModel):
    period: str
    median_salary_mid: float | None
    p25_salary_mid: float | None
    p75_salary_mid: float | None
    vacancy_count: int
    salary_count: int
    employer_count: int


class ForecastResponse(BaseModel):
    granularity: str
    source_dataset: str
    region: str
    occupation_group: str
    input_period: str
    target_period: str
    baseline_salary: float
    catboost_salary: float
    forecast_salary: float
    predicted_change_percent: float
    q4_to_q1_correction_applied: bool
    prediction_strategy: str
