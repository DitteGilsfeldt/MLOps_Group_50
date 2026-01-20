from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from group50.model import EmotionModel
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

MODEL_PATH = "../models/emotion_m.pth"
classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model
    print("Starting up your application as fast as possible, please wait a moment.")
    model = EmotionModel(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    yield

    print("Cleaning up! Like my flatmates never did...")
    del model


app = FastAPI(title="Emotion Recognition API", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(data: UploadFile = File(...)):
    """Predict emotion from an uploaded image."""
    if data.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="YOUR IMAGE IS INVALID!!! PICK ANOTHER ONE >:C")

    image = Image.open(data.file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    return {"emotion": classes[pred_idx.item()], "confidence": confidence.item()}
