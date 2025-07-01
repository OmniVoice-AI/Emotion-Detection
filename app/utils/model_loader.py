from transformers import pipeline

from app.configs.label_to_emojy import label_to_emoji

MODEL_PATH = "models/emotion_detection"


def predict_emotion(text: str):
    sentiment_analysis = pipeline("text-classification", model=MODEL_PATH)
    
    # Analyze emotion
    result = sentiment_analysis(text)[0]
    label = result["label"].capitalize()
    emoji = label_to_emoji.get(label, "❓")
    
    return label, result["score"], emoji
