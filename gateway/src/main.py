"""
Unified API Gateway — single entry point for all inference requests.

Routes to edge (Jetson) or cloud (Triton) based on query complexity,
determined by the orchestrator's routing logic.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Inference Gateway", version="0.1.0")


class InferenceRequest(BaseModel):
    messages: list[dict]
    model: str = "auto"
    stream: bool = False


@app.get("/health")
async def health():
    return {"status": "ok", "service": "gateway"}


@app.post("/v1/chat/completions")
async def chat_completions(req: InferenceRequest):
    # TODO P2: call orchestrator to determine routing (edge vs cloud)
    # TODO P2: forward to appropriate backend
    raise HTTPException(501, "routing not yet implemented")
