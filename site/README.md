# Сайт аналитики рынка труда

Веб-часть проекта отделена от пайплайна подготовки данных и моделирования.

```text
site/
  backend/    # API: данные, фильтры и прогноз CatBoost
  frontend/   # дашборд на React
```

## Локальный запуск

### Backend

```powershell
cd site\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Проверка API: `http://127.0.0.1:8000/api/health`.

### Frontend

```powershell
cd site\frontend
npm install
npm run dev
```

Vite выведет адрес интерфейса в терминал. Открывать его следует по адресу
`http://127.0.0.1:5173`, а не через `localhost`: это устойчивее при активном
VPN или браузерном прокси.

## Границы ответственности

- Backend читает parquet-файлы из `data/processed/` и модель из `models/`.
- Frontend получает только JSON-ответы API.
- Frontend не читает parquet-файлы и не загружает CatBoost-модель напрямую.
