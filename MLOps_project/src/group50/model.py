import timm
import torch
import torch.nn as nn


class EmotionModel(nn.Module):
    """
    A PyTorch module for emotion classification using a pretrained backbone.
    This model leverages the TIMM library to instantiate a ResNet architecture
    adapted for a specific number of emotion classes.

    Attributes:
        backbone: The neural network architecture used for feature extraction.
    """

    def __init__(self, model_name="resnet18", num_classes=7, pretrained=True):
        """
        Initializes the EmotionModel.
        Args:
            model_name: The name of the model architecture from the timm library.
            num_classes: The number of output classes (emotions).
            pretrained: Whether to load ImageNet pretrained weights.
        """
        super().__init__()
        # Load the pretrained backbone
        # in_chans=1 are grayscale but 3 is safer for standard ResNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 3 or x.shape[2] != 48 or x.shape[3] != 48:
            raise ValueError("Expected each sample to have shape [3, 48, 48]")

        return self.backbone(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # The dataset has 7 categories
    model = EmotionModel(num_classes=7).to(device)
    print(f"Model architecture: \n{model}")
    # Test dummy image (Batch, Channels, H, W)
    dummy_input = torch.randn(1, 3, 48, 48).to(device)
    output = model(dummy_input)
    print(f"Output device: {output.device}")  # This can be GPU, CPU or mps
    print(f"Output shape: {output.shape}")  # Size should be [1, 7]
