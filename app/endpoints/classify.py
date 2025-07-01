import uuid
import time
import json
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path

from app.utils.model_loader import predict_emotion

router = APIRouter()

LOG_FILE = Path("data/results.json")
LOG_FILE.parent.mkdir(exist_ok=True)

@router.post("/")
async def classify_emotion(text: str = Form(...)):
    start = time.time()
    text_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    # Classificazione dell'emozione dal testo
    label, score, emoji = predict_emotion(text)

    # Logging su file
    record = {
        "uuid": text_id,
        "text": text,
        "emotion": label,
        "score": score,
        "emoji": emoji,
        "submission_time": now
    }

    existing = []
    if LOG_FILE.exists():
        try:
            existing = json.loads(LOG_FILE.read_text())
        except Exception:
            pass
    existing.append(record)
    LOG_FILE.write_text(json.dumps(existing, indent=2))

    elapsed = (time.time() - start) * 1000  # ms

    return JSONResponse({
        "uuid": text_id,
        "text": text,
        "emotion": label,
        "score": score,
        "emoji": emoji,
        "submission_time": now,
        "execution_time_ms": round(elapsed, 2)
    })
