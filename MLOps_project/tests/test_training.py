import os

import pytest
import torch
import hydra

from hydra import initialize, compose
from group50.data import emotion_data
from group50.model import EmotionModel
from omegaconf import DictConfig
from pathlib import Path

from group50.train import _train_impl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
device = torch.device("mps" if torch.mps.is_available() else "cpu")
DATA_PATH = "data/processed"

REQUIRED_FILES = [
    "data/processed/train_images.pt",
    "data/processed/train_target.pt",
    "data/processed/test_images.pt",
    "data/processed/test_target.pt",
]


@pytest.mark.skipif(not all(os.path.exists(p) for p in REQUIRED_FILES), reason="Processed data files not found")


# KALD PÅ VORES EGEN
# GEM MODEL DER HEDDER NOGET VI SELV VÆLGER
# TEST EVAL SKAL KUNNE EVALERE RESULTET FRA DEN HER PY FIL

def test_training_pipeline():
    with initialize(version_base=None, config_path="../config/testings"):
        config = compose(config_name="train_test1_conf")

        loss_stats = _train_impl(config, use_wandb=False, max_batches=2)

        model_path = PROJECT_ROOT / "models" / "emotion_test.pth"
        
        assert len(loss_stats) >= 2
        assert all(torch.isfinite(torch.tensor(loss_stats)))
        assert model_path.exists()

