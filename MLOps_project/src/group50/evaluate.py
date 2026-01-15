import torch
import timm
from group50.data import emotion_data
from group50.model import EmotionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate_model(model_checkpoint: str) -> None:
    """
    Function to evaluate the model on the test dataset, by calculating accuracy.
    
    Args:
        model_checkpoint: Path to the model checkpoint where weights are saved.
    """
    
    print('Evaluating model')
    print(model_checkpoint)

    model = EmotionModel.to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE)) # loads the model weights
    model.eval()

    _, testset = emotion_data()
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32) # creates the torch dataloader from testset

    correct, total = 0, 0 # Count of correct and total predictions
    
    with torch.no_grad():
        for img, target in test_loader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            output = model(img)
            total += target.size(0)
            correct += (output.argmax(dim=1) == target).sum().item() # correct if output = target prediction
        print(f"Test accuracy: {100 * correct / total:.2f}%")

