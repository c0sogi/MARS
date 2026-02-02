import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import ConvBiGRU


def generate_submission(
    model_path=Config.MODEL_SAVE_PATH,
    output_path="./submission/submission.csv",
    device=Config.DEVICE,
):
    """
    Loads the trained model, performs inference on the test set, and saves the submission file.

    Args:
        model_path (str): Path to the saved model weights.
        output_path (str): Path to save the generated submission CSV.
        device (torch.device): Device to run inference on.
    """
    # 1. Setup
    set_seed(Config.SEED)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Using device: {device}")

    # 2. Data Loading
    # We only need the test loader for inference
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        load_cached_data=Config.LOAD_CACHED_DATA, debug=Config.DEBUG
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = ConvBiGRU()
    model.to(device)

    # 4. Load Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    # 5. Inference Loop
    model.eval()
    preds_list = []
    ids_list = []

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass: Output shape (Batch, 107, 5)
            outputs = model(inputs)

            # Slice to scored length (first 68 positions)
            # Shape becomes: (Batch, 68, 5)
            outputs = outputs[:, : Config.PRED_LEN, :]

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    # 6. Post-Processing
    # Concatenate all batches -> (N_samples, 68, 5)
    preds_arr = np.concatenate(preds_list, axis=0)

    n_samples = preds_arr.shape[0]
    seq_len = preds_arr.shape[1]  # Should be 68

    # Flatten predictions to (N_samples * 68, 5)
    preds_flat = preds_arr.reshape(-1, Config.OUTPUT_DIM)

    # Generate id_seqpos column
    # Repeat each ID 68 times: [id1, id1, ..., id2, id2, ...]
    ids_repeated = np.repeat(ids_list, seq_len)

    # Tile positions 0..67 N times: [0, 1, ..., 67, 0, 1, ..., 67]
    pos_tiled = np.tile(np.arange(seq_len), n_samples)

    # Combine into strings
    id_seqpos = [f"{i}_{p}" for i, p in zip(ids_repeated, pos_tiled)]

    # 7. Create DataFrame and Save
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    df_sub = pd.DataFrame(preds_flat, columns=target_cols)
    df_sub.insert(0, "id_seqpos", id_seqpos)

    print(f"Saving submission to {output_path}...")
    df_sub.to_csv(output_path, index=False)
    print("Submission generated successfully.")
