from fastapi import FastAPI
from app.endpoints import classify
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="EmotionDetection API")

uploads_path = Path("uploads")
uploads_path.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(classify.router, prefix="/classify", tags=["Classify"])
