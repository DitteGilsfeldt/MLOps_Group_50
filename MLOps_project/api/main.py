import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from group50.model import EmotionModel
from PIL import Image, ImageStat
from pydantic import BaseModel
from torchvision import transforms

MODEL_PATH = "../models/emotion_model.pth"
DATABASE_PATH = "../data/database.db"
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


def init_database():
    """Init database for storage of predictions and sample properties"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_emotion TEXT,
            confidence REAL,
            brightness REAL,
            contrast REAL
        )
        """
    )
    conn.commit()
    conn.close()


def calculate_image_properties(image: Image.Image) -> dict:
    """Calculate basic properties of the image.
    Args:
      image: PIL Image object"""
    stat = ImageStat.Stat(image)
    brightness = np.mean(stat.mean)
    contrast = np.mean(stat.stddev)

    return {"brightness": brightness, "contrast": contrast}


async def add_to_database(timestamp: str, predicted_emotion: str, confidence: float, properties: dict):
    """Function to add prediction and image properties to database
    Args:
      predicted_emotion: The emotion predicted by the model
      confidence: The confidence of the prediction
      image: PIL Image object"""

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (timestamp, predicted_emotion, confidence, brightness, contrast)
        VALUES (?, ?, ?, ?, ?)
        """,
        (timestamp, predicted_emotion, confidence, properties["brightness"], properties["contrast"]),
    )
    print(
        f"Added prediction to database: {timestamp}, {predicted_emotion}, {confidence}, {properties['brightness']}, {properties['contrast']}"
    )

    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model
    print("Starting up your application as fast as possible, please wait a moment.")
    init_database()
    model = EmotionModel(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    yield

    print("Cleaning up! Like my flatmates never did...")
    del model


app = FastAPI(title="Emotion Recognition API", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(data: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Predict emotion from an uploaded image."""
    if data.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="YOUR IMAGE IS INVALID!!! PICK ANOTHER ONE >:C")

    image = Image.open(data.file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_emotion = classes[pred_idx.item()]
    confidence = confidence.item()

    now = str(datetime.now())
    properties = calculate_image_properties(image)

    background_tasks.add_task(add_to_database, now, predicted_emotion, confidence, properties)

    return {"emotion": predicted_emotion, "confidence": confidence}
