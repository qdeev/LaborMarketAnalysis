"""Generic normalization helpers."""

import re
from typing import Any


NULL_PLACEHOLDERS = {
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
    "ничего не выбрано",
    "з/п не указана",
    "зарплата не указана",
    "требования не предъявляются",
}

REGION_ALIASES = {
    "адыгея": "Республика Адыгея",
    "республика адыгея": "Республика Адыгея",
    "республика адыгея (адыгея)": "Республика Адыгея",
    "москва": "Москва",
    "г москва": "Москва",
    "город москва": "Москва",
    "санкт-петербург": "Санкт-Петербург",
    "г санкт-петербург": "Санкт-Петербург",
    "город санкт-петербург": "Санкт-Петербург",
    "севастополь": "Севастополь",
    "г севастополь": "Севастополь",
    "ханты-мансийский автономный округ - югра": "Ханты-Мансийский автономный округ - Югра",
    "ханты-мансийский ао - югра": "Ханты-Мансийский автономный округ - Югра",
    "ямало-ненецкий автономный округ": "Ямало-Ненецкий автономный округ",
    "ямало-ненецкий ао": "Ямало-Ненецкий автономный округ",
}

OCCUPATION_KEYWORDS = [
    ("IT и разработка", ("программист", "разработчик", "developer", "devops", "инженер по тестированию", "qa", "data scientist", "аналитик данных")),
    ("Аналитика", ("аналитик", "analysis", "bi ")),
    ("Продажи", ("менеджер по продаж", "продавец", "кассир", "торговый представитель")),
    ("Медицина", ("врач", "медицин", "фельдшер", "медсест", "фармацевт")),
    ("Образование", ("учитель", "преподаватель", "педагог", "воспитатель")),
    ("Производство", ("слесарь", "токарь", "оператор стан", "монтажник", "сварщик")),
    ("Логистика и транспорт", ("водитель", "курьер", "логист", "экспедитор")),
    ("Строительство", ("строител", "прораб", "инженер пто", "мастер смр")),
    ("HoReCa", ("повар", "официант", "бариста", "администратор ресторана")),
    ("Администрирование", ("администратор", "секретарь", "офис-менеджер")),
]


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Normalizer requires pandas for dataframe operations.") from exc
    return pd


def _clean_text(value: Any) -> Any:
    pd = _require_pandas()
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    if text.casefold() in NULL_PLACEHOLDERS:
        return pd.NA
    return text


def _is_missing_value(value: Any) -> bool:
    pd = _require_pandas()
    cleaned = _clean_text(value)
    return bool(pd.isna(cleaned))


class Normalizer:
    """Normalize common placeholders, types, and categorical fields."""

    def normalize_nulls(self, df: Any, columns: list[str] | None = None) -> Any:
        """Convert common placeholder values to null."""
        pd = _require_pandas()
        out = df.copy()
        target_columns = list(out.columns) if columns is None else columns

        for column in target_columns:
            if column not in out.columns:
                continue
            series = out[column]
            if pd.api.types.is_string_dtype(series) or series.dtype == "object":
                out[column] = series.map(_clean_text)

        return out

    def normalize_zero_to_null(self, df: Any, columns: list[str]) -> Any:
        """Convert zero values to null in selected numeric columns."""
        pd = _require_pandas()
        out = df.copy()

        for column in columns:
            if column not in out.columns:
                continue
            numeric = pd.to_numeric(out[column], errors="coerce")
            out[column] = out[column].mask(numeric.eq(0), pd.NA)

        return out

    def normalize_region(self, value: Any) -> Any:
        """Normalize a raw region value to the canonical region label."""
        cleaned = _clean_text(value)
        if _require_pandas().isna(cleaned):
            return {"raw": value, "normalized": None, "is_unknown": True}

        text = str(cleaned)
        normalized_key = (
            text.casefold()
            .replace("ё", "е")
            .replace(".", " ")
            .replace(",", " ")
        )
        normalized_key = re.sub(r"\([^)]*\)", " ", normalized_key)
        normalized_key = re.sub(r"\s+", " ", normalized_key).strip()
        if normalized_key in REGION_ALIASES:
            normalized = REGION_ALIASES[normalized_key]
            return {
                "raw": value,
                "normalized": normalized,
                "is_unknown": False,
            }
        normalized_key = re.sub(r"^(обл|область|край|республика|респ)\s+", "", normalized_key)
        normalized_key = normalized_key.strip()

        if normalized_key in REGION_ALIASES:
            normalized = REGION_ALIASES[normalized_key]
        elif " область" in text.casefold() or text.casefold().endswith(" край"):
            normalized = text
        elif text.casefold().startswith("республика "):
            normalized = text
        else:
            normalized = REGION_ALIASES.get(text.casefold(), text)

        return {
            "raw": value,
            "normalized": normalized,
            "is_unknown": False,
        }

    def normalize_occupation(self, value: Any, source_dataset: str | None = None) -> Any:
        """Normalize a raw occupation/title/professional role."""
        cleaned = _clean_text(value)
        if _require_pandas().isna(cleaned):
            return {
                "raw": value,
                "normalized": None,
                "occupation_group": "unknown",
                "source_dataset": source_dataset,
            }

        text = str(cleaned)
        folded = text.casefold()
        group = "Другое"
        for candidate, keywords in OCCUPATION_KEYWORDS:
            if any(keyword in folded for keyword in keywords):
                group = candidate
                break

        return {
            "raw": value,
            "normalized": text,
            "occupation_group": group,
            "source_dataset": source_dataset,
        }

    def normalize_experience(self, value: Any, source_dataset: str | None = None) -> Any:
        """Normalize source-specific experience values."""
        cleaned = _clean_text(value)
        if _require_pandas().isna(cleaned):
            return {
                "raw": value,
                "experience_min_years": None,
                "experience_max_years": None,
                "experience_group": "unknown",
            }

        text = str(cleaned)
        folded = text.casefold()
        min_years: float | None = None
        max_years: float | None = None

        if folded in {"noexperience", "нет опыта", "без опыта", "0"}:
            min_years, max_years = 0.0, 0.0
        elif folded == "between1and3" or "от 1 года до 3 лет" in folded:
            min_years, max_years = 1.0, 3.0
        elif folded == "between3and6" or "от 3 до 6" in folded:
            min_years, max_years = 3.0, 6.0
        elif folded == "morethan6" or "более 6" in folded:
            min_years, max_years = 6.0, None
        elif folded == "between0and1" or "до 1 года" in folded:
            min_years, max_years = 0.0, 1.0
        else:
            numbers = [float(number.replace(",", ".")) for number in re.findall(r"\d+(?:[,.]\d+)?", folded)]
            if len(numbers) >= 2:
                min_years, max_years = numbers[0], numbers[1]
            elif len(numbers) == 1:
                min_years, max_years = numbers[0], numbers[0]

        if min_years == 0 and max_years == 0:
            group = "no_experience"
        elif min_years is not None and min_years < 1 and (max_years is None or max_years <= 1):
            group = "up_to_1"
        elif min_years is not None and min_years < 3:
            group = "1_to_3"
        elif min_years is not None and min_years < 6:
            group = "3_to_6"
        elif min_years is not None:
            group = "6_plus"
        else:
            group = "unknown"

        return {
            "raw": value,
            "experience_min_years": min_years,
            "experience_max_years": max_years,
            "experience_group": group,
            "source_dataset": source_dataset,
        }

    def normalize_schedule(self, value: Any, source_dataset: str | None = None) -> Any:
        """Normalize source-specific schedule values."""
        cleaned = _clean_text(value)
        if _require_pandas().isna(cleaned):
            return {
                "raw": value,
                "schedule_type": "unknown",
                "is_remote": False,
                "is_shift": False,
                "is_fly_in_fly_out": False,
                "source_dataset": source_dataset,
            }

        text = str(cleaned)
        folded = text.casefold()
        is_remote = any(token in folded for token in ("remote", "удален", "дистанц"))
        is_shift = any(token in folded for token in ("смен", "shift", "график"))
        is_fly = any(token in folded for token in ("вахт", "flyinflyout"))

        if is_fly:
            schedule_type = "fly_in_fly_out"
        elif is_remote:
            schedule_type = "remote"
        elif "полный" in folded or folded == "fullday":
            schedule_type = "full_day"
        elif "неполный" in folded or "part" in folded:
            schedule_type = "part_time"
        elif "гибк" in folded or "flexible" in folded:
            schedule_type = "flexible"
        elif is_shift:
            schedule_type = "shift"
        else:
            schedule_type = "other"

        return {
            "raw": value,
            "schedule_type": schedule_type,
            "is_remote": is_remote,
            "is_shift": is_shift,
            "is_fly_in_fly_out": is_fly,
            "source_dataset": source_dataset,
        }
