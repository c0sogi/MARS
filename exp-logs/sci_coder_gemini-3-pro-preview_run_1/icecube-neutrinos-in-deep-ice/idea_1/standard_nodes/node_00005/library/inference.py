import os
import torch
from library.config import Config
from library.model import TemporalCNN, predict_and_submit
from library.data import get_test_dataloader


def generate_submission():
    """
    Loads the best trained model state, performs inference on the test set,
    and generates the submission CSV file.

    This function:
    1. Initializes the TemporalCNN model.
    2. Loads weights from the best checkpoint.
    3. Loads the test dataset.
    4. Calls the prediction utility to generate and save results.
    """
    # 1. Setup Device and Paths
    device = Config.DEVICE
    weights_path = os.path.join(Config.IDEA_DIR, "best_model.pth")

    # 2. Initialize Model
    print("Initializing TemporalCNN model...")
    model = TemporalCNN()

    # 3. Load Weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Please ensure the model has been trained before running inference."
        )

    print(f"Loading model weights from {weights_path}...")
    # Load state dict, mapping to the correct device
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Load Test Data
    print("Loading test data...")
    test_loader = get_test_dataloader()

    # 5. Generate Predictions and Submit
    # predict_and_submit handles the inference loop, vector-to-angle conversion,
    # and saving the DataFrame to Config.SUBMISSION_PATH.
    predict_and_submit(model, test_loader, device)
