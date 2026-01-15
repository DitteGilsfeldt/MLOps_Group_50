import os

import pytest
import torch
from group50.data import emotion_data

DATA_PATH = "data/processed"

REQUIRED_FILES = [
    "data/processed/train_images.pt",
    "data/processed/train_target.pt",
    "data/processed/test_images.pt",
    "data/processed/test_target.pt",
]


@pytest.mark.skipif(not all(os.path.exists(p) for p in REQUIRED_FILES), reason="Processed data files not found")

# @pytest.mark.skipif(
#     not os.path.exists(DATA_PATH),
#     reason="Data files not found"
# )

def test_data():
    train, test = emotion_data()

    # sizes: your dataset depends on how many images you have, so don't assert 30000/5000 here
    assert len(train) > 0, "Training set is empty"
    assert len(test) > 0, "Test set is empty"

    for dataset in [train, test]:
        x, y = dataset[0]
        assert x.shape == (3, 48, 48), "Input shape incorrect"
        assert y.item() in range(7), "Label out of range (expected 0..6)"

    train_targets = torch.unique(train.tensors[1])
    assert train_targets.min().item() >= 0
    assert train_targets.max().item() <= 6
