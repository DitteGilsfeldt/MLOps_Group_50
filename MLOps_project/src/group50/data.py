import torch
import os
import typer
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# map folder names to class indices
classes_to_idx = {cls: i for i, cls in enumerate(classes)}

# We do not use this, since we use ResNet which expects mean = [0.485, 0.456, 0.406] and std  = [0.229, 0.224, 0.225] normalization instead.
# def normalize(images: torch.Tensor) -> torch.Tensor:
#     """Normalize images."""
#     return (images - images.mean()) / images.std()


def load_data(split_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load images and targets from the specified directory and returns them as tensors."""
    images, targets = [], []
    # 1 channel like MNIST from the course, fixed size (48x48) like described and [0,1] format.
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])])
    
    for class_name in classes:
        class_dir = split_dir / class_name
        label = classes_to_idx[class_name]

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing directory: {class_dir}")

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                with Image.open(img_path) as img:
                    img = transform(img)
                images.append(img)
                targets.append(label)

    if not images:
        raise RuntimeError(f"No images found in {split_dir}")
        
    images_tensor = torch.stack(images) # TO DO: create PyTest to see if [N, 3, 48, 48] is satisfied
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    
    return images_tensor, targets_tensor

def preprocess_data() -> None:
    """Process raw data and save it to processed directory."""
    raw_dir = DATA_ROOT
    processed_dir = DATA_ROOT / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_target = load_data(raw_dir / "train")
    test_images, test_target = load_data(raw_dir / "test")

    # Normalize images

    torch.save(train_images, processed_dir / "train_images.pt")
    torch.save(train_target, processed_dir / "train_target.pt")
    torch.save(test_images, processed_dir / "test_images.pt")
    torch.save(test_target, processed_dir / "test_target.pt")


def emotion_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for emotion recognition."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    preprocess_data()
