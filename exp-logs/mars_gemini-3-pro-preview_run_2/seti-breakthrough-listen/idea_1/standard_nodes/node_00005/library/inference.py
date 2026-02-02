import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.dataset import SETIDataset
from library.model import TechnosignatureResNet


def generate_submission(
    model_path=None,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Generates predictions for the test set using the trained model and saves them to a CSV file.

    Args:
        model_path (str, optional): Path to the trained model weights (.pth).
                                    Defaults to 'best_model.pth' in the working directory.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker processes for data loading.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"Generating submission using model: {model_path}")
    print(f"Output path: {output_path}")

    # 2. Prepare Data
    # We load the metadata dataframe separately to ensure we have the correct IDs in order
    test_metadata_df = pd.read_csv(Config.TEST_METADATA)

    test_dataset = SETIDataset(metadata_path=Config.TEST_METADATA)

    # shuffle=False is critical to match predictions with IDs
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Load Model
    model = TechnosignatureResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    print("Starting inference...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            # Move to CPU and flatten
            probs_np = probs.cpu().numpy().flatten()
            all_probs.append(probs_np)

    # Concatenate all batches
    if all_probs:
        final_predictions = np.concatenate(all_probs)
    else:
        final_predictions = np.array([])

    # 5. Create Submission File
    # Ensure the number of predictions matches the number of test samples
    if len(final_predictions) != len(test_metadata_df):
        raise ValueError(
            f"Mismatch between number of predictions ({len(final_predictions)}) "
            f"and number of test samples ({len(test_metadata_df)})."
        )

    # Update the target column with predictions
    test_metadata_df["target"] = final_predictions

    # Select only the required columns
    submission_df = test_metadata_df[["id", "target"]]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
