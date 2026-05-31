"""
api.py — FastAPI server for the Quantinel PM dashboard.

Serves the dashboard static files and exposes pipeline JSON endpoints consumed
by dashboard/dashboard.js:
  GET /api/pipeline         — latest cached payload (runs pipeline on first call)
  GET /api/pipeline/refresh — force a fresh pipeline run
  GET /api/health           — server + pipeline status

Run:
  set -a; source .env; set +a
  .venv/bin/uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pipeline_service import get_pipeline_payload, pipeline_status

ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "dashboard"

app = FastAPI(title="Quantinel PM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", **pipeline_status()}


@app.get("/api/pipeline")
def pipeline():
    try:
        return get_pipeline_payload(refresh=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/pipeline/refresh")
def pipeline_refresh():
    try:
        return get_pipeline_payload(refresh=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=DASHBOARD, html=True), name="dashboard-static")
