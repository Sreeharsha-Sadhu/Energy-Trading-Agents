from fastapi import FastAPI
from src.api.routes import router as api_router

app = FastAPI(title="Energy Trading Agent Backend")

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "healthy"}
