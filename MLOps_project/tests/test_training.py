import os

import pytest
import torch
import hydra

from hydra import initialize, compose
from group50.data import emotion_data
from group50.model import EmotionModel
from omegaconf import DictConfig

from group50.train import train


device = torch.device("mps" if torch.mps.is_available() else "cpu")
DATA_PATH = "data/processed"

REQUIRED_FILES = [
    "data/processed/train_images.pt",
    "data/processed/train_target.pt",
    "data/processed/test_images.pt",
    "data/processed/test_target.pt",
]


@pytest.mark.skipif(not all(os.path.exists(p) for p in REQUIRED_FILES), reason="Processed data files not found")


def test_training_pipeline():
    with initialize(version_base=None, config_path="../config/tests"):
        config: DictConfig = compose(config_name="train_test1_conf")

        model = EmotionModel(num_classes=7, pretrained=False).to(device)

        train_set, _ = emotion_data()
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config.hyperparameters.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparameters.lr)

        losses = []
        for _ in range(config.hyperparameters.epochs):
            model.train()
            epoch_loss = 0.0
            for img, target in train_dataloader:
                img, target = img.to(device), target.to(device)
                optimizer.zero_grad()
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(train_dataloader))

        assert losses[1] < losses[0]
