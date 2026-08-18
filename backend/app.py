import base64
import binascii
import io
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from knowledge import write_up
from labels import pretty
from model import load_model, predict

app = FastAPI(title="LeafLens CNN", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    imageDataUrl: str = Field(..., description="data:image/...;base64,...")
    notes: str | None = Field(default=None, max_length=500)


class Prediction(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    disease: str
    confidence: str
    healthy: bool
    plant: str
    symptoms: list[str]
    causes: list[str]
    treatment: list[str]
    prevention: list[str]
    summary: str
    model: str
    trained: bool
    topK: list[Prediction]


def decode_data_url(data_url: str) -> Image.Image:
    if not data_url.startswith("data:image/"):
        raise HTTPException(status_code=422, detail="Expected an image data URL")
    _, _, payload = data_url.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Malformed base64 image") from exc
    if len(raw) > 8_000_000:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        return Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Unreadable image") from exc


def bucket(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


@app.get("/health")
def health() -> dict[str, object]:
    _, classes, trained = load_model()
    return {
        "status": "ok",
        "architecture": "FINALCNN (PyTorch)",
        "classes": len(classes),
        "trained": trained,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(body: PredictRequest) -> PredictResponse:
    image = decode_data_url(body.imageDataUrl)
    top, trained = predict(image)
    best_label, best_prob = top[0]
    plant, disease, healthy = pretty(best_label)
    text = write_up(disease, healthy)

    if healthy:
        summary = f"This {plant.lower()} leaf looks healthy ({best_prob:.0%} confidence)."
    else:
        summary = (
            f"The CNN classifies this as {disease.lower()} on {plant.lower()} "
            f"with {best_prob:.0%} confidence."
        )
    if not trained:
        summary += " No trained checkpoint is loaded, so treat this as a placeholder."

    return PredictResponse(
        disease=disease,
        confidence=bucket(best_prob),
        healthy=healthy,
        plant=plant,
        symptoms=text["symptoms"],
        causes=text["causes"],
        treatment=text["treatment"],
        prevention=text["prevention"],
        summary=summary,
        model="FINALCNN (PyTorch)",
        trained=trained,
        topK=[Prediction(label=label, probability=prob) for label, prob in top],
    )
