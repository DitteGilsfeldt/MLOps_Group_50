import logging
import shutil
from pathlib import Path

import torch

import wandb
from group50.data import emotion_data
from group50.model import EmotionModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

log = logging.getLogger(__name__)


def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 10, model_name: str = "emotion_model", wb=True):
    """Train a model on emtion_data.
    Args:
        config: Hydra configuration object containing hyperparameters.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if wb:
        wandb.init(
            project="MLOps_Group_50_Emotion_Recognition",
            entity="zilverwood-dtu",
            config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
            name=f"{model_name}_bs{batch_size}_lr{lr}_ep{epochs}",
        )

    log.info("Time to get that summer body! We are training now!")
    log.info(f"{lr=}, {batch_size=}, {epochs=}")

    model = EmotionModel().to(DEVICE)
    train_set, test_set = emotion_data()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_accuracy = 0.0

    loss_stats = []

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

            if epoch == 0 and i == 0:
                best_loss = loss.item()
                best_accuracy = accuracy

            elif loss.item() < best_loss:
                best_loss = loss.item()
                best_accuracy = accuracy
                save_checkpoint(model, model_name)

            if wb:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "best_loss": best_loss,
                        "train_accuracy": accuracy,
                        "best_accuracy": best_accuracy,
                    }
                )

            if i % 100 == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}, accuracy: {accuracy*100}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for img, target in test_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                output = model(img)
                val_loss += loss_fn(output, target).item()
                val_correct += (output.argmax(dim=1) == target).float().sum().item()
                val_total += target.size(0)

        val_accuracy = val_correct / val_total
        val_loss /= len(test_dataloader)

        if wb:
            wandb.log({"validation_loss": val_loss, "validation_accuracy": val_accuracy})

        loss_stats.append(running_loss / len(train_dataloader))

        log.info("\n -------------------------------------------------------- \n")
        log.info(f"Epoch {epoch} completed. Avg loss: {running_loss / len(train_dataloader)}")
        log.info("\n -------------------------------------------------------- \n")

    log.info("Big summer body done, strong and lean now!")

    # Clean up local wandb directory
    if wb:
        shutil.rmtree(PROJECT_ROOT / "wandb", ignore_errors=True)
        shutil.rmtree("wandb", ignore_errors=True)

    return loss_stats


def save_checkpoint(model, model_name):
    """Function to save model checkpoint

    Args:
        model: The model to save.
        model_name: The name to use for the saved model file.
    """
    torch.save(model.state_dict(), DATA_ROOT / f"{model_name}.pth")


if __name__ == "__main__":
    train()
