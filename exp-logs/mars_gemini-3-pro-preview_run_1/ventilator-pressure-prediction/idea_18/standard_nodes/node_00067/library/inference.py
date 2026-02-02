import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import prepare_data
from library.model import VentilatorModel
from library.trainer import set_seed


def generate_predictions(config: Config, load_cached_data: bool = True):
    """
    Generates predictions for the test set using the trained model and saves
    the submission file.

    Args:
        config (Config): Configuration object containing paths and hyperparameters.
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
    """
    # 1. Setup
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Starting inference on device: {device}")

    # 2. Prepare Data
    # This handles feature engineering, scaling (using saved scaler), and reshaping
    print("Preparing test dataset...")
    test_dataset = prepare_data(config, split="test", load_cached_data=load_cached_data)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Initialize Model
    print("Initializing model...")
    model = VentilatorModel(config).to(device)

    # 4. Load Weights
    model_path = os.path.join(config.WORKING_DIR, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. Inference Loop
    all_preds = []
    all_ids = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            inputs = batch["input"].to(device)
            u_out = batch["u_out"].to(device)
            ids = batch["ids"]  # Keep IDs on CPU until needed

            # Forward pass
            # Model returns (final_pred, aux_pred). We only need final_pred.
            final_pred, _ = model(inputs, u_out=u_out)

            # Squeeze the last dimension if it exists: (Batch, Seq, 1) -> (Batch, Seq)
            if final_pred.dim() == 3:
                final_pred = final_pred.squeeze(-1)

            # Collect results (move to CPU)
            all_preds.append(final_pred.cpu().numpy())
            all_ids.append(ids.numpy())

    # 6. Post-Processing
    # Concatenate all batches: Result is (Total_Breaths, 80)
    preds_array = np.concatenate(all_preds, axis=0)
    ids_array = np.concatenate(all_ids, axis=0)

    # Flatten to (Total_Rows,)
    flat_preds = preds_array.flatten()
    flat_ids = ids_array.flatten()

    # 7. Create Submission DataFrame
    submission_df = pd.DataFrame({"id": flat_ids, "pressure": flat_preds})

    # Ensure output directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Save
    print(f"Saving submission to {submission_path}...")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission generated successfully. Shape: {submission_df.shape}")
