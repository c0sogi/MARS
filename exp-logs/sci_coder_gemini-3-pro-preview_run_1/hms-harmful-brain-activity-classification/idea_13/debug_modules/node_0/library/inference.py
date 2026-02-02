import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.models import AuxiliaryFusionNet
from library.data import MultiModalDataset
from library.utils import seed_everything


def predict_and_submit(
    model_path=Config.MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    metadata_path=Config.TEST_CSV,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    debug_limit=None,
):
    """
    Runs inference on the test set and generates a submission file.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        output_path (str): Path where the submission CSV will be saved.
        metadata_path (str): Path to the test metadata CSV.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cpu' or 'cuda').
        debug_limit (int, optional): Limit the number of test samples for debugging.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Starting inference using model: {model_path}")
    print(f"Device: {device}")

    # 2. Load Data
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {metadata_path}")

    test_df = pd.read_csv(metadata_path)

    if debug_limit:
        test_df = test_df.iloc[:debug_limit]
        print(f"Debug mode: Limiting inference to {len(test_df)} samples.")

    # Initialize Dataset and DataLoader
    # We use augment=False for deterministic inference
    test_ds = MultiModalDataset(test_df, mode="test", augment=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    # 3. Load Model
    model = AuxiliaryFusionNet()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found at {model_path}")

    # Load state dict
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for i, (eeg, spec) in enumerate(test_loader):
            eeg = eeg.to(device)
            spec = spec.to(device)

            # Forward pass
            # The model returns (joint_logits, aux_eeg_logits, aux_spec_logits)
            # We only use the joint_logits for the final submission
            joint_logits, _, _ = model(eeg, spec)

            # Apply Softmax to get probabilities summing to 1
            probs = F.softmax(joint_logits, dim=1)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

    # 5. Post-processing
    if len(all_probs) == 0:
        print("Warning: No predictions generated.")
        return pd.DataFrame()

    predictions = np.concatenate(all_probs, axis=0)

    # Ensure we have the correct number of predictions matching the dataframe
    if len(predictions) != len(test_df):
        print(
            f"Warning: Prediction count ({len(predictions)}) differs from metadata count ({len(test_df)}). Truncating/Aligning."
        )
        min_len = min(len(predictions), len(test_df))
        predictions = predictions[:min_len]
        test_df = test_df.iloc[:min_len]

    # 6. Construct Submission DataFrame
    # Columns must be: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote
    submission = pd.DataFrame()
    submission["eeg_id"] = test_df["eeg_id"]

    # Map predictions to columns based on Config.CLASS_NAMES order
    # CLASS_NAMES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
    submission["seizure_vote"] = predictions[:, 0]
    submission["lpd_vote"] = predictions[:, 1]
    submission["gpd_vote"] = predictions[:, 2]
    submission["lrda_vote"] = predictions[:, 3]
    submission["grda_vote"] = predictions[:, 4]
    submission["other_vote"] = predictions[:, 5]

    # 7. Save Submission
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return submission
