FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user site/backend/requirements.txt site/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r site/backend/requirements.txt

COPY --chown=user site/backend site/backend
COPY --chown=user data/processed/ml_broad_monthly_salary_dataset.parquet data/processed/ml_broad_monthly_salary_dataset.parquet
COPY --chown=user data/processed/ml_broad_quarterly_salary_dataset.parquet data/processed/ml_broad_quarterly_salary_dataset.parquet
COPY --chown=user data/processed/broad_monthly_modeling_setup.json data/processed/broad_monthly_modeling_setup.json
COPY --chown=user data/processed/broad_quarterly_modeling_setup.json data/processed/broad_quarterly_modeling_setup.json
COPY --chown=user models/catboost_broad_monthly_residual_salary_model.cbm models/catboost_broad_monthly_residual_salary_model.cbm
COPY --chown=user models/catboost_broad_quarterly_residual_salary_model.cbm models/catboost_broad_quarterly_residual_salary_model.cbm

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--app-dir", "site/backend", "--host", "0.0.0.0", "--port", "7860"]
