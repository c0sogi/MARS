import torch
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import HFSI_BiLSTM


def generate_submission(config: Config):
    """
    Loads the trained model, performs inference on the test set,
    and generates the submission CSV file.

    Args:
        config (Config): Configuration object containing file paths and hyperparameters.
    """
    # 1. Setup
    seed_everything(config.SEED)
    device = get_device()

    print("Initializing inference pipeline...")

    # 2. Data Loading
    # We use get_dataloaders to ensure consistent preprocessing and caching with the training phase.
    # We unpack the tuple to get the test_loader and test_ids.
    # train_loader and val_loader are not needed for inference.
    _, _, test_loader, test_ids = get_dataloaders(config)

    # Determine input dimension from a sample batch in test_loader
    # This ensures the model architecture matches the data features dynamically.
    try:
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch["input"].shape[-1]
    except StopIteration:
        raise ValueError("Test loader is empty. Cannot determine input dimension.")

    print(f"Data loaded. Input dimension: {input_dim}")

    # 3. Model Initialization
    model = HFSI_BiLSTM(config, input_dim).to(device)

    # Load the best checkpoint
    if not os.path.exists(config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {config.MODEL_CHECKPOINT}"
        )

    print(f"Loading model weights from {config.MODEL_CHECKPOINT}...")
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    # 4. Inference Loop
    test_preds_list = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)

            # Forward pass
            # Output shape: (Batch, Seq_Len, 1)
            preds = model(inputs)

            # Squeeze to (Batch, Seq_Len) and move to CPU/Numpy for storage
            test_preds_list.append(preds.squeeze(-1).cpu().numpy())

    # 5. Post-processing
    # Concatenate all batches: (N_Test_Breaths, 80)
    test_preds_arr = np.concatenate(test_preds_list)

    # Flatten predictions and IDs to 1D arrays to match submission format
    test_preds_flat = test_preds_arr.flatten()
    test_ids_flat = test_ids.flatten()

    # Ensure lengths match (sanity check)
    if len(test_preds_flat) != len(test_ids_flat):
        raise ValueError(
            f"Mismatch between predictions ({len(test_preds_flat)}) and IDs ({len(test_ids_flat)}) length"
        )

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids_flat, "pressure": test_preds_flat})

    # Sort by ID to ensure correct order as required by competition format
    # (Although test_ids should already be ordered by breath_id then time_step)
    submission_df.sort_values(by="id", inplace=True)

    # 7. Save Submission
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    print("Submission generation complete.")
