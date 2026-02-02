import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, get_device, ensure_directories
from library.dataset import get_dataloaders
from library.model import SiameseDebertaWithScalars


def predict(debug: bool = False):
    """
    Loads the trained model, performs inference on the test set, and generates
    a submission CSV file.

    Args:
        debug (bool): If True, runs on a subset of the data for testing purposes.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    ensure_directories()

    print(f"Starting inference (Debug={debug})...")

    # 2. Data Loading
    # get_dataloaders returns (train, val, test). We only need test.
    _, _, test_loader = get_dataloaders(debug=debug)

    # Load the test metadata to get the IDs.
    # We need to ensure the IDs match the order of the DataLoader.
    # The DataLoader is created with shuffle=False, so order is preserved.
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Apply the same slicing logic as ChatbotDataset if in debug mode
    if debug:
        test_df = test_df.head(50)

    ids = test_df["id"].values

    # 3. Model Loading
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = SiameseDebertaWithScalars()

    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Please train the model first."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    print("Running prediction loop...")
    with torch.no_grad():
        for batch in tqdm(
            test_loader, disable=True
        ):  # Disable tqdm for cleaner logs as requested
            # Move inputs to device
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)

            # Forward pass
            logits = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                scalar_features=scalar_features,
            )

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    if len(all_probs) > 0:
        final_probs = np.vstack(all_probs)
    else:
        final_probs = np.zeros((0, 3))

    # 5. Submission Generation
    # Check consistency
    if len(ids) != len(final_probs):
        raise ValueError(
            f"Mismatch between number of IDs ({len(ids)}) and predictions ({len(final_probs)})."
        )

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "winner_model_a": final_probs[:, 0],
            "winner_model_b": final_probs[:, 1],
            "winner_tie": final_probs[:, 2],
        }
    )

    print(f"Saving submission to {Config.SUBMISSION_SAVE_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
    print("Inference complete.")
