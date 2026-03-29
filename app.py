import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict

MODEL_PATH = os.path.join("saved_models", "abuse_model.joblib")

app = FastAPI(
    title="GuardAI Abuse Detection API",
    description="API for detecting SAFE, MILD, or ABUSIVE messages",
    version="1.0.0"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

class MessageRequest(BaseModel):
    message: str = Field(..., example="Thank you for helping me today!")

class MessageResponse(BaseModel):
    message: str
    label: str
    confidence: float
    scores: Dict[str, float]

@app.get("/")
def root():
    return {"message": "GuardAI API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=MessageResponse)
def predict(payload: MessageRequest):
    try:
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        prediction = model.predict([payload.message])[0]
        probabilities = model.predict_proba([payload.message])[0]
        classes = model.classes_

        score_map = {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(classes, probabilities)
        }

        confidence = round(max(probabilities) * 100, 2)

        return MessageResponse(
            message=payload.message,
            label=prediction,
            confidence=confidence,
            scores=score_map
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))