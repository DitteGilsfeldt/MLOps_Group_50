import torch
import typer
from pathlib import Path
from datetime import datetime

from group50.data import emotion_data
from group50.model import EmotionModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 1) -> None:
    """Train a model on emtion_data."""

    print("Time to get that summer body! We are training now...")
    print(f"{lr=}, {batch_size=}, {epochs=}")
 
    model = EmotionModel().to(DEVICE)
    train_set, _ = emotion_data()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            total = target.size(0)
            correct = (y_pred.argmax(dim=1) == target).float().sum().item()
            accuracy = correct / total

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}, accuracy: {accuracy*100}%")
        print("\n -------------------------------------------------------- \n")
        print(f"Epoch {epoch} completed. Avg loss: {running_loss / len(train_dataloader)}")
        print("\n -------------------------------------------------------- \n")
    
    print("Big summer body done, saving model...")
    torch.save(model.state_dict(), DATA_ROOT / f"emotion_model_{datetime.now()}.pth")
    return None


if __name__ == "__main__":
    train()
