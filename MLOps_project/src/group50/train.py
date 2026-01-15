import torch
import typer
import hydra
import logging
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig


from group50.data import emotion_data
from group50.model import EmotionModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="training_conf")

def train(config: DictConfig) -> None:
    """Train a model on emtion_data."""

    lr = config.hyperparameters.lr
    batch_size = config.hyperparameters.batch_size
    epochs = config.hyperparameters.epochs

    log.info("Time to get that summer body! We are training now!")
    log.info(f"{lr=}, {batch_size=}, {epochs=}")
 
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
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}, accuracy: {accuracy*100}%")            

        log.info("\n -------------------------------------------------------- \n")
        log.info(f"Epoch {epoch} completed. Avg loss: {running_loss / len(train_dataloader)}")
        log.info("\n -------------------------------------------------------- \n")
    
    log.info("Big summer body done, strong and lean now!")
    torch.save(model.state_dict(), DATA_ROOT / f"emotion_model.pth")
    return None


if __name__ == "__main__":
    train()
