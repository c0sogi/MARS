import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.dataset import get_test_dataloader
from library.model import RNAModel


def generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Generates the submission file for the RNA degradation prediction task.

    This function:
    1. Loads the test dataset and the best trained model.
    2. Performs inference on the full 107-length sequences.
    3. Maps the 3 predicted targets to the 5 required submission columns.
    4. Fills unscored columns (deg_pH10, deg_50C) with 0.0.
    5. Saves the result to ./submission/submission.csv.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        batch_size (int): Batch size for inference.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Generating submission on device: {device}")

    # 2. Load Data
    print("Loading test data...")
    test_loader = get_test_dataloader(
        load_cached_data=load_cached_data,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Load Model
    print(f"Loading model from {Config.MODEL_PATH}...")
    # Initialize model structure
    model = RNAModel(config=Config)

    # Check if model weights exist
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Please train the model first."
        )

    # Load weights
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_preds = []
    all_ids = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)

            # Get IDs (list of strings)
            batch_ids = batch["id"]

            # Forward pass
            # Output shape: (Batch, Seq_Len, 3)
            # The 3 channels correspond to Config.TARGET_COLS:
            # ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
            outputs = model(seq, loop, dist)

            # Move to CPU and numpy
            preds = outputs.cpu().numpy()

            all_preds.append(preds)
            all_ids.extend(batch_ids)

    # 5. Post-processing
    # Concatenate all predictions: (N_Samples, 107, 3)
    final_preds = np.concatenate(all_preds, axis=0)
    num_samples = final_preds.shape[0]
    seq_len = final_preds.shape[1]

    print(f"Processed {num_samples} samples with sequence length {seq_len}.")

    # Flatten predictions to (N_Samples * 107, 3) for dataframe construction
    flat_preds = final_preds.reshape(-1, 3)

    # Generate id_seqpos column
    # We need to repeat each ID 107 times and append the sequence position index
    # Repeat IDs: [id1, id1, ..., id2, id2, ...]
    repeated_ids = np.repeat(all_ids, seq_len)

    # Tile indices: [0, 1, ..., 106, 0, 1, ...]
    tiled_indices = np.tile(np.arange(seq_len), num_samples)

    # Create id_seqpos strings
    # Format: id_sequence_position (e.g., id_00b436dec_0)
    id_seqpos = [f"{rid}_{idx}" for rid, idx in zip(repeated_ids, tiled_indices)]

    # 6. Create Submission DataFrame
    # The submission requires 5 columns + id_seqpos.
    # Model predicts: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"] (indices 0, 1, 2)
    # Unscored columns: ["deg_pH10", "deg_50C"] (fill with 0.0)

    submission_df = pd.DataFrame()
    submission_df["id_seqpos"] = id_seqpos

    # Map predictions
    submission_df["reactivity"] = flat_preds[:, 0]
    submission_df["deg_Mg_pH10"] = flat_preds[:, 1]
    submission_df["deg_pH10"] = 0.0  # Unscored, fill with 0
    submission_df["deg_Mg_50C"] = flat_preds[:, 2]
    submission_df["deg_50C"] = 0.0  # Unscored, fill with 0

    # 7. Save Files
    # Ensure the ./submission directory exists as per task requirements
    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")

    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)

    # Also save to the working directory defined in Config for record keeping
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Saving copy to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Submission generation complete.")
