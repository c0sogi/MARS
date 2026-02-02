import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import GLiClassModel
from library.data import get_test_loader


def predict_and_submit(
    model_path=Config.MODEL_SAVE_PATH, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set using the Independent-Instance 2.5D Volumetric Ensemble (I2VE)
    strategy and saves the submission file.

    Args:
        model_path (str): Path to the trained model weights.
        output_path (str): Path where the submission CSV will be saved.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Prepare Data
    # get_test_loader initializes GLiClassDataset with split='test'.
    # The dataset class handles caching internally: checks for .npy files in working dir,
    # loads them if present, or processes raw DICOMs and saves to cache if missing.
    print("Initializing Test Loader...")
    test_loader = get_test_loader()

    # 3. Load Model
    print(f"Loading model from {model_path}...")
    # We set pretrained=False because we are loading specific trained weights
    model = GLiClassModel(
        backbone=Config.BACKBONE,
        pretrained=False,
        num_classes=Config.NUM_CLASSES,
        in_chans=Config.IN_CHANNELS,
    )

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model file not found at {model_path}. Using random weights.")

    model = model.to(device)
    model.eval()

    # 4. Inference Loop
    print("Starting inference on test data...")
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, _, subject_ids in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Collect results (move to CPU and flatten)
            all_probs.append(probs.cpu().numpy().flatten())
            all_ids.append(subject_ids.numpy().flatten())

    # 5. Aggregation
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs)
        all_ids = np.concatenate(all_ids)
    else:
        all_probs = np.array([])
        all_ids = np.array([])

    # Create temporary DataFrame with slice-level predictions
    df_slices = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # Aggregate slice predictions to subject predictions (Mean Consensus)
    df_pred = df_slices.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

    # 6. Ensure Completeness
    # Load the original test metadata to get the full list of required Subject IDs.
    # This handles cases where a subject might have been skipped during processing
    # (e.g., due to missing files or empty scans) by filling with 0.5.
    if os.path.exists(Config.TEST_METADATA_PATH):
        df_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        required_ids = df_meta["BraTS21ID"].unique()
        df_required = pd.DataFrame({"BraTS21ID": required_ids})

        # Merge predictions into the required list
        final_submission = df_required.merge(df_pred, on="BraTS21ID", how="left")

        # Fill missing predictions with 0.5 (random guess for missing data)
        final_submission["MGMT_value"] = final_submission["MGMT_value"].fillna(0.5)
    else:
        final_submission = df_pred

    # 7. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_submission.to_csv(output_path, index=False)

    print(f"Inference complete. Submission saved to {output_path}")
    print("Sample predictions:")
    print(final_submission.head())
