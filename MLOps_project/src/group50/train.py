import torch
import typer

from group50.data import emotion_data
from group50.model import EmotionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 2) -> None:
    """Train a model on emtion_data."""

    print("Time to get that summer body! We are training now...")
    print(f"{lr=}, {batch_size=}, {epochs=}")
 
    model = EmotionModel().to(DEVICE)
    train_set, _ = emotion_data()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    return None



if __name__ == "__main__":
    train()
