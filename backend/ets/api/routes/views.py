# -*- coding: utf-8 -*-
"""
Views router.
Serves the MLflow-like Data-Dense Web Dashboard for ETS from a clean, separated frontend file.
"""
from __future__ import annotations

import logging
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the single-page application dashboard directly from the frontend directory."""
    base_dir = os.path.abspath(".")
    frontend_path = os.path.join(base_dir, "frontend", "index.html")
    if not os.path.exists(frontend_path):
        # Check parent directory fallback (if run from backend/ directory)
        frontend_path = os.path.join(os.path.dirname(base_dir), "frontend", "index.html")
        
    if not os.path.exists(frontend_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Frontend index.html not found. Please ensure it exists in: {frontend_path}"
        )
    
    try:
        with open(frontend_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as exc:
        logger.error("Failed to read frontend file: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to load frontend dashboard: {exc}")
