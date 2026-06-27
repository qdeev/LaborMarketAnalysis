"""Shared constants for preprocessing raw vacancy sources."""

from pathlib import Path
from typing import Final, Literal, cast


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

SOURCE_TRUDVSEM_LATEST: Final[str] = "trudvsem_latest"
SOURCE_HH_GITHUB: Final[str] = "hh_github"
SOURCE_HH_KAGGLE: Final[str] = "hh_kaggle"
SOURCE_COMBINED: Final[str] = "combined"

SourceDataset = Literal[
    "trudvsem_latest",
    "hh_github",
    "hh_kaggle",
    "combined",
]

SUPPORTED_SOURCE_DATASETS: Final[tuple[SourceDataset, ...]] = (
    SOURCE_TRUDVSEM_LATEST,
    SOURCE_HH_GITHUB,
    SOURCE_HH_KAGGLE,
    SOURCE_COMBINED,
)

RAW_DATASET_PATHS: Final[dict[SourceDataset, Path]] = {
    SOURCE_TRUDVSEM_LATEST: RAW_DATA_DIR / "vacancies_trudvsem_21_05_26.csv",
    SOURCE_HH_GITHUB: RAW_DATA_DIR / "vacancies_hh_github.sqlite",
    SOURCE_HH_KAGGLE: RAW_DATA_DIR / "vacancies_hh_kaggle.csv",
    SOURCE_COMBINED: RAW_DATA_DIR / "vacancies_hh_trudvsem_mendeley.csv",
}

SOURCE_DATASET_LABELS: Final[dict[SourceDataset, str]] = {
    SOURCE_TRUDVSEM_LATEST: "Trudvsem latest snapshot",
    SOURCE_HH_GITHUB: "HeadHunter GitHub SQLite",
    SOURCE_HH_KAGGLE: "HeadHunter Kaggle CSV",
    SOURCE_COMBINED: "HH/Trudvsem/Mendeley combined CSV",
}


def validate_source_dataset(source_dataset: str) -> SourceDataset:
    """Return a supported source dataset name or raise a clear error."""
    if source_dataset not in SUPPORTED_SOURCE_DATASETS:
        supported = ", ".join(SUPPORTED_SOURCE_DATASETS)
        raise ValueError(
            f"Unsupported source_dataset {source_dataset!r}. "
            f"Expected one of: {supported}."
        )
    return cast(SourceDataset, source_dataset)
