import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join("saved_models", "abuse_model.joblib")

app = FastAPI(
    title="GuardAI Abuse Detection API",
    description="API for checking whether a title and description are good or bad",
    version="2.0.0"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


class MessageRequest(BaseModel):
    title: str = Field(..., example="Community update")
    description: str = Field(..., example="Thank you for helping me today!")


class MessageResponse(BaseModel):
    status: str = Field(..., example="good")


@app.get("/")
def root():
    return {"message": "GuardAI API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=MessageResponse)
def predict(payload: MessageRequest):
    try:
        title = payload.title.strip()
        description = payload.description.strip()

        if not title and not description:
            raise HTTPException(
                status_code=400,
                detail="Both title and description cannot be empty"
            )

        # Combine title + description into one text input for the model
        full_text = f"{title} {description}".strip()

        # Predict using the existing model
        prediction = str(model.predict([full_text])[0]).strip().upper()

        # SAFE -> good
        # everything else -> bad
        if prediction == "SAFE":
            return MessageResponse(status="good")
        else:
            return MessageResponse(status="bad")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))