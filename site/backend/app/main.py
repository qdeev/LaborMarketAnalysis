"""FastAPI entry point for the labour-market analytics website."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.analytics import router as analytics_router


app = FastAPI(
    title="Labour Market Analytics API",
    version="0.1.0",
)

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
configured_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*default_origins, *configured_origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analytics_router)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Return a lightweight readiness signal for the frontend."""
    return {"status": "ok"}
