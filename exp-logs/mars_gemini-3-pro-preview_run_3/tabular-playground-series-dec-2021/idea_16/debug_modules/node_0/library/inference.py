import os
import torch
import pandas as pd
import numpy as np
from library.config import Config, TrainConfig
from library.utils import get_device
from library.data import get_dataloaders
from library.model import ParallelLowRankDCNResNet


def predict(
    model_path=Config.MODEL_SAVE_PATH,
    batch_size=TrainConfig.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Loads the trained model, performs inference on the test set,
    and saves the predictions to a CSV file in the required format.

    Args:
        model_path (str): Path to the saved model state dict.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        output_path (str): Path to save the submission CSV.
    """
    device = get_device()
    print(f"Inference using device: {device}")

    # 1. Load Data
    # get_dataloaders returns (train_loader, val_loader, test_loader, test_ids)
    # We only need the test components here.
    print("Loading test data...")
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers, load_cached_data=True
    )

    # 2. Load Model
    print(f"Loading model architecture and weights from {model_path}...")
    model = ParallelLowRankDCNResNet()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please ensure the model is trained."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Inference Loop
    predictions = []
    print("Starting inference...")

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predicted class (0-indexed)
            # Argmax is sufficient; Softmax is monotonic and not needed for class selection
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())

    # 4. Post-processing
    # Convert 0-indexed predictions (0-6) back to 1-indexed targets (1-7)
    # The dataset documentation specifies Cover_Type values are integers starting from 1.
    predictions = np.array(predictions) + 1

    # Verify lengths match to ensure data integrity
    if len(predictions) != len(test_ids):
        raise ValueError(
            f"Mismatch in lengths: Predictions ({len(predictions)}) vs IDs ({len(test_ids)})"
        )

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # 6. Save Submission
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print("First 5 rows of submission:")
    print(submission_df.head())
