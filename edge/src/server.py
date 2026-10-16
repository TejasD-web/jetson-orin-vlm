"""
Edge inference microservice — runs on the Jetson Orin Nano.

Wraps llama-server (llama.cpp) in a FastAPI service with:
- OpenAI-compatible chat/completions endpoint
- SSE streaming support
- Single-inference concurrency control (Jetson memory constraint)
- Prometheus metrics export
- Health/readiness probes for K8s
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import ChatCompletionRequest, ChatCompletionResponse
from .config import Settings


settings = Settings()

# Single-inference semaphore — Jetson has limited unified memory
_inference_sem = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify llama-server is reachable. Shutdown: cleanup."""
    # TODO P1: health-check llama-server, load model metadata
    app.state.model_name = settings.model_name
    app.state.start_time = time.time()
    yield
    # cleanup if needed


app = FastAPI(
    title="Edge Inference Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": getattr(app.state, "model_name", settings.model_name)}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
            	"id": getattr(app.state, "model_name", settings.model_name),                
		"object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Routes to local llama-server with concurrency control.
    """
    async with _inference_sem:
        # TODO P1: forward to llama-server, handle streaming vs non-streaming
        raise HTTPException(501, "inference forwarding not yet implemented")
