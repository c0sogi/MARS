import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import SiameseDeberta
from library.data_processing import get_dataloaders
from library.utils import seed_everything


def generate_predictions(load_cached_data=True):
    """
    Generates predictions for the test set using the trained model and saves
    the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Reproducibility
    seed_everything(Config.seed)

    print(f"Starting inference on device: {Config.device}")

    # 2. Load Data
    # We only need the test_loader
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    model = SiameseDeberta()
    model.to(Config.device)

    if os.path.exists(Config.model_save_path):
        print(f"Loading model weights from {Config.model_save_path}")
        state_dict = torch.load(Config.model_save_path, map_location=Config.device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.model_save_path}. Please train the model first."
        )

    model.eval()

    # 4. Inference Loop
    all_probs = []

    print("Running prediction loop...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids_a = batch["input_ids_a"].to(Config.device)
            attention_mask_a = batch["attention_mask_a"].to(Config.device)
            input_ids_b = batch["input_ids_b"].to(Config.device)
            attention_mask_b = batch["attention_mask_b"].to(Config.device)
            scalar_features = batch["scalar_features"].to(Config.device)

            # Forward pass
            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)

    # 5. Prepare Submission
    # Load test metadata to get IDs. The DataLoader is sequential (shuffle=False),
    # so the order matches the CSV.
    test_df = pd.read_csv(Config.test_path)

    # Handle debug mode where test_df might be larger than predictions if dataloader was subsampled internally
    # (Though get_dataloaders handles subsampling of the dataframe itself in debug mode)
    if Config.debug:
        test_df = test_df.head(len(all_probs))

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": all_probs[:, 0],
            "winner_model_b": all_probs[:, 1],
            "winner_tie": all_probs[:, 2],
        }
    )

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Save
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print(f"Submission shape: {submission_df.shape}")
    print("Head of submission:")
    print(submission_df.head())
