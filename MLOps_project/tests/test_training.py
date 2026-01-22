import os
from pathlib import Path

import pytest
import torch
from group50.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
DATA_PATH = "data/processed"

REQUIRED_FILES = [
    "data/processed/train_images.pt",
    "data/processed/train_target.pt",
    "data/processed/test_images.pt",
    "data/processed/test_target.pt",
]


@pytest.mark.skipif(not all(os.path.exists(p) for p in REQUIRED_FILES), reason="Processed data files not found")
def test_training_pipeline():
    loss_stats = train(lr=0.001, batch_size=32, epochs=2, model_name="emotion_test", wb=False, workers=0)

    model_path = PROJECT_ROOT / "models" / "emotion_test.pth"

    assert len(loss_stats) >= 2
    assert all(torch.isfinite(torch.tensor(loss_stats)))
    assert model_path.exists()
