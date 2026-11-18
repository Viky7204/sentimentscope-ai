"""
Prediction module for SentimentScope AI
"""

import os
import joblib
from preprocess import full_clean


LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

EMOJI_MAP = {
    "Negative": "😞",
    "Neutral": "😐",
    "Positive": "😊"
}


class SentimentPredictor:
    def __init__(self, model_path: str = "models/sentiment_model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Please run: python app/train.py"
            )
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> dict:
        """
        Predict sentiment of the given text.
        Returns label, confidence, and probabilities.
        """
        if not text or not str(text).strip():
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "message": "Please enter some text."
            }

        cleaned = full_clean(text)
        if not cleaned:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "message": "Text became empty after cleaning."
            }

        pred = self.model.predict([cleaned])[0]
        proba = self.model.predict_proba([cleaned])[0]

        label = LABEL_MAP.get(pred, "Unknown")
        confidence = round(float(max(proba)) * 100, 1)

        return {
            "label": label,
            "emoji": EMOJI_MAP.get(label, ""),
            "confidence": confidence,
            "proba_negative": round(float(proba[0]) * 100, 1),
            "proba_neutral": round(float(proba[1]) * 100, 1),
            "proba_positive": round(float(proba[2]) * 100, 1),
            "cleaned_text": cleaned
        }
