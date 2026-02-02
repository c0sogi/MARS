import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    TEST_METADATA_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    NUM_FOLDS,
    SEED,
)
from library.utils import seed_everything
from library.dataset import SlabDataset, get_transforms
from library.model import WITSNetwork


def inference_fn(load_cached_data=True):
    """
    Runs the inference pipeline:
    1. Loads test metadata and prepares the SlabDataset (WITS-II pipeline).
    2. Loads trained models from all available folds.
    3. Generates predictions for each slab.
    4. Aggregates predictions by Subject ID (Mean of 3 slabs * N models).
    5. Saves the submission file.
    """
    seed_everything(SEED)

    # 1. Load Metadata
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    test_metadata = pd.read_csv(TEST_METADATA_PATH)
    print(f"Loaded test metadata: {len(test_metadata)} subjects.")

    # 2. Prepare Dataset
    # We use 'val' transforms (just normalization/tensor conversion) for inference
    test_dataset = SlabDataset(
        test_metadata,
        transform=get_transforms("val"),
        load_cached_data=load_cached_data,
        split_name="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test dataset ready. Total slabs to predict: {len(test_dataset)}")

    # 3. Load Models and Predict
    # We will store predictions in a list of arrays, then average them
    # Shape: (Num_Samples, Num_Models)
    all_model_preds = []

    model = WITSNetwork()
    model.to(DEVICE)

    models_found = 0

    for fold in range(NUM_FOLDS):
        model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model checkpoint for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model for Fold {fold}...")
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            models_found += 1
        except Exception as e:
            print(f"Error loading fold {fold}: {e}")
            continue

        # Run inference for this fold
        fold_preds = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(DEVICE)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_preds.append(probs)

        # Concatenate batches for this fold
        fold_preds = np.concatenate(fold_preds, axis=0).flatten()  # Shape (N,)
        all_model_preds.append(fold_preds)

    if models_found == 0:
        print("Error: No trained models found. Generating dummy 0.5 predictions.")
        # Fallback for debugging/pipeline verification without trained models
        final_slab_preds = np.full(len(test_dataset), 0.5)
    else:
        # Stack predictions from all models: Shape (N, M)
        all_model_preds = np.stack(all_model_preds, axis=1)
        # Average across models
        final_slab_preds = np.mean(all_model_preds, axis=1)

    # 4. Aggregate by Subject ID
    # The dataset stores the subject ID for each slab in self.ids
    # Since shuffle=False, these align with final_slab_preds
    slab_ids = test_dataset.ids

    df_preds = pd.DataFrame({"BraTS21ID": slab_ids, "prob": final_slab_preds})

    # Group by Subject ID and take the mean of the 3 slabs
    submission_df = df_preds.groupby("BraTS21ID")["prob"].mean().reset_index()
    submission_df.rename(columns={"prob": "MGMT_value"}, inplace=True)

    # 5. Format and Save
    # Ensure ID format matches requirements (5-digit string or integer)
    # The sample submission uses integers in the dataframe but visualizes as strings.
    # We keep them as integers as per the provided sample_submission.csv schema description.

    # Ensure all subjects in metadata are in submission (fill missing with 0.5 if any dropped)
    # Though SlabDataset should handle all valid subjects.
    expected_ids = test_metadata["BraTS21ID"].unique()
    missing_ids = np.setdiff1d(expected_ids, submission_df["BraTS21ID"].values)

    if len(missing_ids) > 0:
        print(
            f"Warning: {len(missing_ids)} subjects missing from predictions. Filling with 0.5."
        )
        missing_df = pd.DataFrame({"BraTS21ID": missing_ids, "MGMT_value": 0.5})
        submission_df = pd.concat([submission_df, missing_df], ignore_index=True)

    # Sort by ID
    submission_df.sort_values("BraTS21ID", inplace=True)

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
