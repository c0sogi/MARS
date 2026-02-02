import os
import torch
import pandas as pd
import numpy as np

from library.config import WORKING_DIR, SUBMISSION_PATH, TARGET_COLS, BATCH_SIZE, SEED
from library.model import HybridRNNTransformer
from library.dataset import get_dataloaders


def generate_submission(
    model_path=os.path.join(WORKING_DIR, "best_model.pth"),
    output_path=SUBMISSION_PATH,
    batch_size=BATCH_SIZE,
    num_workers=2,
):
    """
    Generates the submission file by running inference on the test set.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
    """
    # 1. Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference running on device: {device}")

    # 2. Load Test Data
    # get_dataloaders returns (train, val, test). We only need test.
    # It handles caching internally.
    print("Loading test dataloader...")
    _, _, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=batch_size, num_workers=num_workers
    )

    # 3. Initialize Model
    print("Initializing model architecture...")
    model = HybridRNNTransformer().to(device)

    # 4. Load Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model checkpoint not found at {model_path}. Using random initialization."
        )

    # 5. Inference Loop
    model.eval()
    ids_list = []
    preds_list = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            seq = batch["sequence"].to(device)
            struct = batch["structure"].to(device)
            loop = batch["predicted_loop_type"].to(device)
            ids = batch["id"]

            # Forward pass
            # Output shape: (Batch_Size, 107, 5)
            outputs = model(seq, struct, loop)

            # Store results
            ids_list.extend(ids)
            preds_list.append(outputs.cpu().numpy())

    if not preds_list:
        print("No predictions generated. Exiting.")
        return

    # Concatenate all batches: (Total_Samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # 6. Format Predictions
    print("Formatting predictions...")
    submission_rows = []

    # Iterate over each sample
    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 5)

        # Iterate over each position in the sequence (0 to 106)
        for seqpos in range(sample_preds.shape[0]):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            # Construct row dictionary
            row = {
                "id_seqpos": row_id,
                "reactivity": row_values[0],
                "deg_Mg_pH10": row_values[1],
                "deg_pH10": row_values[2],
                "deg_Mg_50C": row_values[3],
                "deg_50C": row_values[4],
            }
            submission_rows.append(row)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Ensure correct column order
    cols = ["id_seqpos"] + TARGET_COLS
    submission_df = submission_df[cols]

    # 7. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission successfully saved to {output_path}")
