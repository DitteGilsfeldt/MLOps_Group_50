import pytest
import torch
from group50.model import EmotionModel


def test_model():
    model = EmotionModel(num_classes=7)
    dummy_input = torch.randn(1, 3, 48, 48)
    output = model(dummy_input)
    assert output.shape == (1, 7)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_error_on_wrong_shape():
    model = EmotionModel(num_classes=7)
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))

    with pytest.raises(ValueError, match=r"Expected each sample to have shape \[3, 48, 48\]"):
        model(torch.randn(1, 3, 48, 49))


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model_parametrized(batch_size: int) -> None:
    model = EmotionModel(num_classes=7)
    x = torch.randn(batch_size, 3, 48, 48)
    y = model(x)
    assert y.shape == (batch_size, 7)
