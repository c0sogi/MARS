import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import RSNADataset
from library.model import FractureMILModel


def generate_submission(config=Config, debug=False, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        config: Configuration class with hyperparameters and paths.
        debug (bool): If True, runs inference on a small subset of the test data.
        load_cached_data (bool): Whether to use cached file paths in the dataset.
    """
    # Set random seeds for reproducibility
    config.seed_everything(config.SEED)

    device = torch.device(config.DEVICE)
    print(f"Running inference on device: {device}")

    # --- 1. Load Model ---
    # Initialize model architecture
    # We use pretrained=False to avoid downloading ImageNet weights during inference
    model = FractureMILModel(pretrained=False)
    model.to(device)

    # Load trained weights
    weights_path = os.path.join(config.CACHE_DIR, "best_model.pth")
    if os.path.exists(weights_path):
        print(f"Loading model weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model weights not found at {weights_path}. Using random initialization."
        )

    model.eval()

    # --- 2. Prepare Data ---
    test_meta_path = config.TEST_METADATA_PATH
    if not os.path.exists(test_meta_path):
        print(f"Error: Test metadata not found at {test_meta_path}")
        return

    test_df = pd.read_csv(test_meta_path)

    if debug:
        print("Debug mode enabled: Processing subset of test data.")
        test_df = test_df.head(config.BATCH_SIZE * 2)

    # Initialize Dataset and DataLoader
    # load_cached_paths=load_cached_data ensures we use the parquet cache if available
    test_dataset = RSNADataset(test_df, config, load_cached_paths=load_cached_data)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Inference Loop ---
    study_ids = test_df["StudyInstanceUID"].tolist()

    # Dictionary to store predictions: {StudyInstanceUID: {target_name: probability}}
    study_preds = {}
    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    print("Starting prediction loop...")

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass: (Batch, Slices, 1, H, W) -> (Batch, 8)
            outputs = model(images)

            # Convert to numpy
            preds_batch = outputs.cpu().numpy()

            # Map predictions to StudyInstanceUIDs
            start_idx = i * config.BATCH_SIZE
            end_idx = start_idx + images.size(0)
            batch_uids = study_ids[start_idx:end_idx]

            for uid, preds in zip(batch_uids, preds_batch):
                study_preds[uid] = {col: float(p) for col, p in zip(target_cols, preds)}

    # --- 4. Format Submission ---
    print("Formatting submission...")

    sample_sub_path = config.SAMPLE_SUBMISSION_PATH
    if not os.path.exists(sample_sub_path):
        print(f"Error: Sample submission not found at {sample_sub_path}")
        return

    submission_df = pd.read_csv(sample_sub_path)

    # Helper to map row_id to probability
    def get_prob(row_id):
        for target in target_cols:
            suffix = f"_{target}"
            if row_id.endswith(suffix):
                # Extract StudyInstanceUID by removing the suffix
                study_uid = row_id[: -len(suffix)]

                # Check if we have a prediction for this study
                if study_uid in study_preds:
                    return study_preds[study_uid][target]

        # Default value for missing predictions (e.g. in debug mode or missing data)
        return 0.5

    # Apply mapping
    submission_df["fractured"] = submission_df["row_id"].apply(get_prob)

    # --- 5. Save Output ---
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
