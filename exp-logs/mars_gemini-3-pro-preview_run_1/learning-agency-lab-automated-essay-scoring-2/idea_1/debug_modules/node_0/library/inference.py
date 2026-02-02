import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DANRegressor


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set and saves them to a submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data for the test set.
    """
    # 1. Set Reproducibility
    seed_everything(Config.SEED)

    # 2. Prepare Data
    # We only need the test_loader here. get_dataloaders handles caching internally.
    print("Loading test data...")
    _, _, test_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Initialize Model
    device = torch.device(Config.DEVICE)
    model = DANRegressor(Config).to(device)

    # 4. Load Trained Weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {Config.MODEL_SAVE_PATH}. "
            "Please run the training pipeline first."
        )

    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Inference Loop
    print("Running inference on test set...")
    all_essay_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            essay_ids = batch["essay_ids"]
            input_ids = batch["input_ids"].to(device)

            # Forward pass
            # Output shape: (batch_size, 1) -> squeeze to (batch_size)
            outputs = model(input_ids).squeeze(-1)

            # Collect results
            all_essay_ids.extend(essay_ids)
            all_preds.extend(outputs.cpu().numpy())

    # 6. Post-processing
    # Convert to numpy array
    preds_np = np.array(all_preds)

    # Clip predictions to valid range [1, 6]
    preds_clipped = np.clip(preds_np, 1, 6)

    # Round to nearest integer
    preds_rounded = np.round(preds_clipped).astype(int)

    # 7. Create Submission DataFrame
    submission_df = pd.DataFrame({"essay_id": all_essay_ids, "score": preds_rounded})

    # 8. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    print(f"Generated predictions for {len(submission_df)} essays.")
