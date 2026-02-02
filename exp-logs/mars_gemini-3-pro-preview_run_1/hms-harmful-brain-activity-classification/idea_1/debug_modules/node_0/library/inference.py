import os
import torch
import pandas as pd
import numpy as np
from library.config import (
    DEVICE,
    CACHE_DIR,
    SUBMISSION_CSV,
    TEST_CSV,
    TARGET_COLS,
    BATCH_SIZE,
    set_seed,
)
from library.model import SpectrogramCRNN
from library.data_loader import get_dataloaders
from library.utils import normalize_probabilities, verify_submission_format


def predict_and_submit(
    model_path: str = None, device: str = DEVICE, batch_size: int = BATCH_SIZE
):
    """
    Loads the best trained model, runs inference on the test set,
    and generates the submission.csv file.

    Args:
        model_path (str): Path to the saved model weights. Defaults to CACHE_DIR/best_model.pth.
        device (str): Compute device ('cpu' or 'cuda').
        batch_size (int): Batch size for inference.
    """
    set_seed()

    # 1. Define Model Path
    if model_path is None:
        model_path = os.path.join(CACHE_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(
            f"Error: Model weights not found at {model_path}. Cannot proceed with inference."
        )
        return

    # 2. Initialize Model
    print(f"Initializing model on {device}...")
    model = SpectrogramCRNN()
    model.to(device)

    # 3. Load Weights
    print(f"Loading weights from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return

    model.eval()

    # 4. Prepare Data
    # We only need the test loader. load_cached_data=True ensures we use the same stats as training.
    _, _, test_loader = get_dataloaders(
        test_batch_size=batch_size, load_cached_data=True
    )

    if test_loader is None:
        print("Error: Failed to create test data loader.")
        return

    # 5. Inference Loop
    print(f"Starting inference on {len(test_loader.dataset)} test samples...")
    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            # Forward pass
            # Model outputs Softmax probabilities
            outputs = model(data)

            all_preds.append(outputs.cpu().numpy())

    # Concatenate all batches
    if not all_preds:
        print("Error: No predictions generated.")
        return

    predictions = np.vstack(all_preds)

    # 6. Post-Processing
    # Ensure probabilities sum to exactly 1.0 per row
    predictions = normalize_probabilities(predictions)

    # 7. Generate Submission DataFrame
    try:
        test_df = pd.read_csv(TEST_CSV)
    except FileNotFoundError:
        print(f"Error: Test metadata file not found at {TEST_CSV}")
        return

    # Sanity check: predictions length must match metadata length
    if len(predictions) != len(test_df):
        print(
            f"Error: Prediction count ({len(predictions)}) does not match test set size ({len(test_df)})."
        )
        return

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=TARGET_COLS)
    submission_df.insert(0, "eeg_id", test_df["eeg_id"])

    # 8. Verify and Save
    if verify_submission_format(submission_df, TARGET_COLS):
        os.makedirs(os.path.dirname(SUBMISSION_CSV), exist_ok=True)
        submission_df.to_csv(SUBMISSION_CSV, index=False)
        print(f"Submission successfully saved to {SUBMISSION_CSV}")

        # Print first few rows for confirmation
        print("\nSubmission Head:")
        print(submission_df.head())
    else:
        print("Submission verification failed. File was not saved.")
