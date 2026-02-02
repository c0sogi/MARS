import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.model import KinematicMLP
from library.dataset import get_dataloaders
from library.train import set_seed


def predict(threshold=0.5, debug=False, load_cached_data=True):
    """
    Generates predictions for the test set using the trained CK-ResNet model.
    Loads the saved model checkpoint, runs inference on the test set, applies
    the decision threshold, and saves the formatted submission file.

    Args:
        threshold (float): The decision threshold for binary classification (0 to 1).
                           Predictions with probability >= threshold are classified as 1.
        debug (bool): If True, processes a small subset of the data for debugging purposes.
        load_cached_data (bool): If True, attempts to load pre-processed features from disk/cache.
                                 If False or cache missing, re-processes the data.

    Returns:
        pd.DataFrame: The final submission dataframe containing 'contact_id' and 'contact'.
    """
    # 1. Setup Environment
    set_seed()
    Config.setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Data
    # get_dataloaders returns (train_loader, val_loader, test_loader).
    # We only require the test_loader for inference.
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        debug=debug,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Initialize Model
    # The input dimensions depend on the feature window size.
    # We infer these dimensions from the first batch of the test loader.
    try:
        sample_inputs, _ = next(iter(test_loader))
        input_dim = sample_inputs.shape[1]

    except StopIteration:
        print("Error: Test loader is empty. Cannot determine model dimensions.")
        return pd.DataFrame(columns=["contact_id", "contact"])

    model = KinematicMLP(input_dim=input_dim)

    # Load trained weights
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        state_dict = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for inputs, contact_ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # Collect results
            all_probs.append(probs.cpu().numpy().flatten())
            all_ids.extend(contact_ids)

    # 5. Post-processing
    if all_probs:
        y_probs = np.concatenate(all_probs)
    else:
        y_probs = np.array([])

    # Apply the optimized threshold
    y_pred = (y_probs >= threshold).astype(int)

    # 6. Generate Submission File
    df_submission = pd.DataFrame({"contact_id": all_ids, "contact": y_pred})

    # Save to disk
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total Predictions: {len(df_submission)}")
    print(f"Positive Predictions: {y_pred.sum()} (Ratio: {y_pred.mean():.4f})")

    return df_submission
