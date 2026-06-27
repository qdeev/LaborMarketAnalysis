# LaborMarketAnalysis

Проект по курсу Big Data в Samsung Academy при Курском государственном университете.

Проект посвящен анализу вакансий и прогнозированию медианной зарплаты по сегментам рынка труда. Сегмент задается регионом, профессиональной группой и временным периодом. В проекте есть полный пайплайн подготовки данных, обучение CatBoost-моделей, backtesting и веб-интерфейс для просмотра аналитики и прогнозов.

## Что сделано

- Собраны и приведены к единой canonical-схеме вакансии из нескольких источников.
- Построены месячные и квартальные агрегаты рынка труда.
- Подготовлены ML-датасеты для прогноза следующей наблюдаемой зарплаты.
- Обучены CatBoost-модели для месячного и квартального горизонта.
- Проведен time-based backtesting.
- Реализован веб-интерфейс с фильтрами, графиком динамики, таблицей периодов и прогнозом.

## Источники данных

В работе использовались открытые датасеты вакансий:

- Trudvsem / Работа России;
- HeadHunter из Kaggle-датасета;
- HeadHunter из GitHub/SQLite-источника;
- объединенный исторический набор вакансий.

Сырые данные не загружены в репозиторий, потому что они занимают несколько гигабайт. В репозитории оставлены только компактные финальные артефакты, необходимые для демонстрации сайта и модели.

## Структура проекта

```text
.
├── data/processed/      # компактные финальные ML-витрины и setup-файлы
├── models/              # финальные CatBoost-модели
├── scripts/             # CLI-скрипты пайплайна, обучения и backtesting
├── site/                # веб-интерфейс и backend API
│   ├── backend/         # FastAPI + DuckDB + CatBoost inference
│   └── frontend/        # React + Vite dashboard
└── src/
    ├── preprocessing/   # чтение, нормализация, canonical-схема, агрегация
    ├── modeling/        # обучение, tuning, evaluation, backtesting
    └── notebooks/       # исследовательские ноутбуки
```

## Данные в репозитории

Для работы сайта и инференса в репозитории оставлены:

```text
data/processed/ml_broad_monthly_salary_dataset.parquet
data/processed/ml_broad_quarterly_salary_dataset.parquet
data/processed/broad_monthly_modeling_setup.json
data/processed/broad_quarterly_modeling_setup.json
models/catboost_broad_monthly_residual_salary_model.cbm
models/catboost_broad_quarterly_residual_salary_model.cbm
```

Большие файлы вроде `data/raw/`, `canonical_vacancies.parquet`, промежуточных parquet и кэшей обучения исключены через `.gitignore`.

## ML-задача

Целевая переменная: следующая наблюдаемая медианная зарплата сегмента.

Для стабильности модель обучалась не напрямую на зарплату, а на изменение относительно текущей медианной зарплаты:

```text
target_log_salary_delta = log(target_salary) - log(current_median_salary_mid)
```

После прогноза значение восстанавливается:

```text
prediction = current_median_salary_mid * exp(predicted_log_delta)
```

Основной финальный вариант:

```text
quarterly residual CatBoost + q4_to_q1_min_catboost_baseline
```


## Результаты

Квартальная модель оказалась стабильнее месячной на backtesting.

| Модель / стратегия | Mean MAPE | Worst MAPE | Mean WAPE | APE p90 |
|---|---:|---:|---:|---:|
| Monthly best blend | 17.55% | 27.89% | 18.06% | 36.24% |
| Quarterly raw CatBoost | 16.29% | 22.76% | 16.55% | 34.15% |
| Quarterly final strategy | 15.19% | 17.46% | 15.80% | 32.73% |

Вывод: при текущем качестве и разреженности данных квартальный прогноз дает более устойчивый результат, а MAPE около 15% можно считать практическим потолком для текущей постановки.

## Локальный запуск сайта

Backend:

```powershell
cd site\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Проверка API:

```text
http://127.0.0.1:8000/api/health
```

Frontend:

```powershell
cd site\frontend
npm install
npm.cmd run dev -- --host 127.0.0.1 --port 5174
```

Открыть:

```text
http://127.0.0.1:5174/
```

## Основные скрипты

Пайплайн источников:

```powershell
python scripts/run_source_pipeline.py
python scripts/run_merge_sources.py
python scripts/run_monthly_aggregation.py
```

Подготовка broad ML-датасетов:

```powershell
python scripts/prepare_broad_monthly_salary_ml_dataset.py
python scripts/prepare_broad_quarterly_salary_ml_dataset.py
```

Обучение финальных residual-моделей:

```powershell
python scripts/train_broad_monthly_residual_catboost.py
python scripts/train_broad_quarterly_residual_catboost.py
```

Backtesting:

```powershell
python scripts/backtest_broad_monthly_residual_catboost.py
python scripts/backtest_broad_quarterly_residual_catboost.py
```

## Веб-интерфейс

Сайт показывает:

- месячный и квартальный режимы;
- фильтры по региону и профессиональной группе;
- KPI по последнему доступному периоду;
- динамику медианной зарплаты и числа вакансий;
- прогноз следующего месяца или квартала;
- таблицу истории выбранного сегмента.

Frontend написан на React/Vite, backend на FastAPI. Backend читает parquet через DuckDB и выполняет инференс CatBoost-моделей.

## Ограничения

- Данные вакансий шумные и неполные: зарплата часто указана диапазоном или отсутствует.
- Источники различаются по структуре и качеству.
- Некоторые регионы и профессиональные группы представлены разреженно.
- Модель прогнозирует агрегированную медианную зарплату сегмента, а не зарплату отдельной вакансии.
- Квартальный прогноз стабильнее, но менее детализирован по времени, чем месячный.

## Ссылка на сайт

https://labor-market-analysis.vercel.app/
