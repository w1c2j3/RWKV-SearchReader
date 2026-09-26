"""Restricted deployment entrypoint; only asynchronous jobs are exposed."""
from fastapi import FastAPI
from .jobs import router

app = FastAPI(title="RWKV SearchReader Jobs", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(router)

@app.get("/healthz")
def health():
    return {"status": "ok"}
