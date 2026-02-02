import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device
from library.model import ParallelDCNResNet
from library.data_loader import get_dataloaders


def load_best_model(input_dim, device):
    """
    Instantiates the ParallelDCNResNet model and loads the best weights saved during training.

    Args:
        input_dim (int): The number of input features.
        device (torch.device): The device to load the model onto.

    Returns:
        model (nn.Module): The model with loaded weights.
    """
    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT,
    ).to(device)

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        print(f"Loading best model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random initialization."
        )

    return model


def generate_submission(model, test_loader, test_ids):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        test_ids (np.ndarray): Array of IDs corresponding to the test data.
    """
    device = get_device()
    model.eval()

    predictions = []
    print("Generating predictions on test set...")

    with torch.no_grad():
        for batch in test_loader:
            # test_loader yields a tuple (X_batch,)
            X_batch = batch[0].to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    # Map predictions back to 1-7 range (model outputs 0-6)
    final_preds = np.array(predictions) + 1

    # Create Submission DataFrame
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: final_preds})

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run_inference(batch_size=Config.BATCH_SIZE, num_workers=4):
    """
    Orchestrates the full inference pipeline: loads data, loads model, and generates submission.

    Args:
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
    """
    device = get_device()

    # 1. Get DataLoaders
    # We only need test_loader and test_ids here.
    # load_cached_data=True ensures we use the pre-processed data if available.
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=True, num_workers=num_workers
    )

    # 2. Determine Input Dimension
    # Fetch a single batch to determine the input feature dimension dynamically
    sample_batch = next(iter(test_loader))
    input_dim = sample_batch[0].shape[1]

    # 3. Load Model
    model = load_best_model(input_dim, device)

    # 4. Generate Submission
    generate_submission(model, test_loader, test_ids)
