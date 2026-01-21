import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# map folder names to class indices
classes_to_idx = {cls: i for i, cls in enumerate(classes)}


def load_data(split_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load images and targets from the specified directory and returns them as tensors."""
    images, targets = [], []
    # 1 channel like MNIST from the course, fixed size (48x48) like described and [0,1] format.
    transform = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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

    images_tensor = torch.stack(images)  # TO DO: create PyTest to see if [N, 3, 48, 48] is satisfied
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
    # This path must match the folder you just created
    cloud_path = "/gcs/group50-emotion-data/processed"
    local_path = "data/processed"

    # Vertex AI will find 'cloud_path' thanks to GCS FUSE
    data_dir = cloud_path if os.path.exists(cloud_path) else local_path

    print(f"Loading data from: {data_dir}")

    # These names match the files I see in your upload
    train_images = torch.load(os.path.join(data_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(data_dir, "train_target.pt"))
    test_images = torch.load(os.path.join(data_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(data_dir, "test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


def dataset_statistics(datadir: str = "data") -> None:
    print("Running dataset statistics")
    print("Data directory:", datadir)

    files = os.listdir(datadir)
    print("Number of files:", len(files))


if __name__ == "__main__":
    preprocess_data()
