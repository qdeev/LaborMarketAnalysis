"""Read-only endpoints for labour-market analytics."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.config import DATASET_CONFIGS
from app.schemas import (
    FilterOptionsResponse,
    ForecastResponse,
    OverviewResponse,
    QuarterlyTrendPoint,
    SegmentHistoryPoint,
    SegmentRow,
)
from app.services.analytics import AnalyticsRepository
from app.services.forecasting import SalaryForecaster


Granularity = Literal["monthly", "quarterly"]

router = APIRouter(prefix="/api", tags=["analytics"])
forecasters = {name: SalaryForecaster(config) for name, config in DATASET_CONFIGS.items()}


def _repository(granularity: Granularity) -> AnalyticsRepository:
    return AnalyticsRepository(DATASET_CONFIGS[granularity])


def _filters(
    source_dataset: str | None,
    region: str | None,
    occupation_group: str | None,
) -> dict[str, str | None]:
    return {
        "source_dataset": source_dataset,
        "region": region,
        "occupation_group": occupation_group,
    }


@router.get("/filters", response_model=FilterOptionsResponse)
def get_filter_options(
    granularity: Granularity = "quarterly",
    source_dataset: str | None = None,
    region: str | None = None,
    occupation_group: str | None = None,
) -> FilterOptionsResponse:
    return FilterOptionsResponse(
        **_repository(granularity).filter_options(_filters(source_dataset, region, occupation_group))
    )


@router.get("/overview", response_model=OverviewResponse)
def get_overview(
    granularity: Granularity = "quarterly",
    source_dataset: str | None = None,
    region: str | None = None,
    occupation_group: str | None = None,
    period: str | None = None,
) -> OverviewResponse:
    result = _repository(granularity).overview(_filters(source_dataset, region, occupation_group), period)
    if result is None:
        raise HTTPException(status_code=404, detail="No matching segments were found")
    return OverviewResponse(**result)


@router.get("/trend", response_model=list[QuarterlyTrendPoint])
def get_trend(
    granularity: Granularity = "quarterly",
    source_dataset: str | None = None,
    region: str | None = None,
    occupation_group: str | None = None,
) -> list[QuarterlyTrendPoint]:
    return [
        QuarterlyTrendPoint(**row)
        for row in _repository(granularity).trend(_filters(source_dataset, region, occupation_group))
    ]


@router.get("/segments", response_model=list[SegmentRow])
def get_segment_rows(
    granularity: Granularity = "quarterly",
    source_dataset: str | None = None,
    region: str | None = None,
    occupation_group: str | None = None,
    period: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[SegmentRow]:
    _, rows = _repository(granularity).segment_rows(
        _filters(source_dataset, region, occupation_group),
        period,
        limit,
    )
    return [SegmentRow(**row) for row in rows]


@router.get("/segment-history", response_model=list[SegmentHistoryPoint])
def get_segment_history(
    occupation_group: str,
    source_dataset: str | None = None,
    region: str | None = None,
    granularity: Granularity = "quarterly",
) -> list[SegmentHistoryPoint]:
    return [
        SegmentHistoryPoint(**row)
        for row in _repository(granularity).segment_history(source_dataset, region, occupation_group)
    ]


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    occupation_group: str,
    source_dataset: str | None = None,
    region: str | None = None,
    period: str | None = None,
    granularity: Granularity = "quarterly",
) -> ForecastResponse:
    row = _repository(granularity).forecast_row(source_dataset, region, occupation_group, period)
    if row is None:
        raise HTTPException(status_code=404, detail="No matching segment was found")
    return ForecastResponse(**forecasters[granularity].predict(row))
