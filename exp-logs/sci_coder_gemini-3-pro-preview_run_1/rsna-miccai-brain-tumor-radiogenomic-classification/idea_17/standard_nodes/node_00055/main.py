"""
Runfile for Glioblastoma MGMT Promoter Methylation Prediction.
Implements AA-WIIS-Net training, validation, failure analysis, and submission.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ensure library modules are importable
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import BraTSDataset, get_transforms
from library.model import AAWIISNet
from library.train import run_training


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Execution Device: {device}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training Phase ===")
    # run_training handles data loading, model init, training loop, and saving best model
    best_model_path = run_training()
    print(f"Best model saved at: {best_model_path}")

    # ==========================================
    # 3. Validation Evaluation
    # ==========================================
    print("\n=== Starting Validation Phase ===")

    # Load the trained model
    # We use pretrained=False because we are loading specific weights immediately after
    # and we want to avoid unnecessary downloads/initialization overhead.
    # The AAWIISNet init handles the architectural changes (weight inflation) correctly.
    model = AAWIISNet(pretrained=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Data
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    val_dataset = BraTSDataset(
        df_val,
        transform=get_transforms("val"),
        is_train=False,
        cache_name="val_roi_eval",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Inference on Validation Set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            # Sigmoid to get probabilities
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    # Aggregate Predictions per Patient
    # The dataset generates 3 slabs per patient in sequential order.
    # We reshape the flat predictions to (Num_Subjects, 3) and take the mean.
    num_slabs = len(Config.SLAB_DEPTHS)

    preds_array = np.array(all_preds)
    targets_array = np.array(all_targets)

    num_subjects = len(df_val)
    if len(preds_array) != num_subjects * num_slabs:
        raise ValueError(
            f"Prediction count {len(preds_array)} does not match expected {num_subjects * num_slabs}"
        )

    preds_reshaped = preds_array.reshape(num_subjects, num_slabs)
    targets_reshaped = targets_array.reshape(num_subjects, num_slabs)

    # Consensus: Mean probability across slabs
    patient_scores = preds_reshaped.mean(axis=1)
    patient_labels = targets_reshaped[
        :, 0
    ]  # Labels are identical for all slabs of a patient

    # Compute Final Metric
    val_auc = roc_auc_score(patient_labels, patient_scores)
    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n=== Starting Failure Analysis ===")

    # Calculate absolute error
    errors = np.abs(patient_labels - patient_scores)

    # Extract metadata features for correlation analysis
    # We use file counts as a proxy for scan resolution/depth
    meta_features = []
    for _, row in df_val.iterrows():
        feats = {}
        # Count files in FLAIR directory as a proxy
        flair_rel_path = row["flair_path"]
        full_path = os.path.join(Config.INPUT_DIR, flair_rel_path)
        try:
            # Fast count
            feats["flair_count"] = len(
                [name for name in os.listdir(full_path) if name.endswith(".dcm")]
            )
        except Exception:
            feats["flair_count"] = 0
        meta_features.append(feats)

    df_analysis = pd.DataFrame(meta_features)
    df_analysis["error"] = errors

    # Compute correlation
    if not df_analysis.empty and "flair_count" in df_analysis.columns:
        correlation = df_analysis["flair_count"].corr(df_analysis["error"])
        print(f"Correlation between Error and FLAIR File Count: {correlation}")
    else:
        print("Could not compute correlation due to missing data.")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.6705454545454544

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = BraTSDataset(
            df_test,
            transform=get_transforms("val"),
            is_train=False,
            cache_name="test_roi",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                test_preds.extend(probs)

        # Aggregate Test Predictions
        test_preds_array = np.array(test_preds)
        num_test_subjects = len(df_test)

        if len(test_preds_array) != num_test_subjects * num_slabs:
            raise ValueError(f"Test prediction count {len(test_preds_array)} mismatch.")

        test_preds_reshaped = test_preds_array.reshape(num_test_subjects, num_slabs)
        test_scores = test_preds_reshaped.mean(axis=1)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": test_scores}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({val_auc}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
