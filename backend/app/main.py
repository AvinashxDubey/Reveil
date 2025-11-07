from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.model_loader import get_model
from app.routers import auth, prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

app = FastAPI(title="Reveil Bot Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(prediction.router)

@app.get("/")
def root():
    return {
        "message": "Reveil Bot Detection API", "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "auth": "/auth",
            "predict": "/predict"
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=settings.debug)