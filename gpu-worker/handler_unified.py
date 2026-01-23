"""
Unified Self-hosted GPU Handler for Dubbing Studio

Placeholder for US-002: FastAPI server with health endpoint.
This file will be expanded to include all AI pipeline endpoints.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Dubbing Studio API",
    description="Unified AI pipeline for video dubbing",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint - will be expanded in US-002."""
    return {"status": "placeholder", "message": "US-002 will implement full health check"}
