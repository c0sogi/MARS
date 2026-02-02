import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import FractureDataset
from library.model import FractureModel
from library.utils import seed_everything


def generate_submission(
    checkpoint_path=os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    debug=Config.DEBUG,
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cpu' or 'cuda').
        debug (bool): If True, runs on a small subset of the test data.
    """
    seed_everything(Config.SEED)

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG Mode: Inference on {len(test_df)} samples.")

    # 2. Prepare Dataset and DataLoader
    # Note: load_cached_data=False for test usually, unless we pre-processed test data similarly.
    # Given the constraints, we'll let the dataset handle caching if it wants,
    # but typically test data might be fresh.
    test_dataset = FractureDataset(test_df, mode="test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    if not os.path.exists(checkpoint_path):
        # If no model is found (e.g. in a fresh run without training), we cannot predict meaningfully.
        # However, to prevent crash during pipeline checks, we might initialize a random model
        # or raise an error. Raising error is safer.
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    model = FractureModel(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # No need to download ImageNet weights, we load our own
        num_classes=Config.NUM_CLASSES,
    )

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    print("Starting inference...")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches: Shape (N_studies, 7)
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
    else:
        all_probs = np.empty((0, Config.NUM_CLASSES))

    # 5. Post-processing and Formatting
    # We need to transform the (N, 7) array into the submission format
    # Format: row_id, fractured
    # row_id examples: '1.2.826.0.1.3680043.10001_C1', '1.2.826.0.1.3680043.10001_patient_overall'

    submission_rows = []
    target_cols = Config.TARGET_COLS  # ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    # Ensure alignment between dataframe rows and predictions
    if len(test_df) != len(all_probs):
        raise ValueError(
            f"Mismatch between metadata rows ({len(test_df)}) and predictions ({len(all_probs)})"
        )

    for idx, row in test_df.iterrows():
        study_uid = row["StudyInstanceUID"]
        probs = all_probs[idx]  # Array of 7 probabilities for C1-C7

        # 1. Add rows for specific vertebrae
        current_study_probs = {}
        for class_idx, col_name in enumerate(target_cols):
            p = float(probs[class_idx])
            current_study_probs[col_name] = p

            row_id = f"{study_uid}_{col_name}"
            submission_rows.append({"row_id": row_id, "fractured": p})

        # 2. Add row for patient_overall
        # Logic: patient_overall is the max probability among C1-C7
        p_overall = max(current_study_probs.values())
        row_id_overall = f"{study_uid}_{Config.OVERALL_COL}"
        submission_rows.append({"row_id": row_id_overall, "fractured": p_overall})

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # 6. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Generated {len(submission_df)} prediction rows.")
