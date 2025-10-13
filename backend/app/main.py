from fastapi import FastAPI
from .core.config import settings

app = FastAPI(
    title="Reveil Bot Detection API",
    description="Bot detection API with JWT authentication",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Reveil Bot Detection API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=settings.debug)