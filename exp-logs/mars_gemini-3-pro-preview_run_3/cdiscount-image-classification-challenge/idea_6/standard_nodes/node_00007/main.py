import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import (
    VAL_FEATURES,
    VAL_LABELS,
    TEST_FEATURES,
    TEST_IDS,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    NUM_CLASSES,
)
from library.metadata import generate_metadata
from library.feature_extractor import run_feature_extraction
from library.training import train_ensemble
from library.data_utils import FeatureDataset, get_category_encoder


def main():
    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # 1. Generate Metadata
    print("Step 1: Generating Metadata...")
    generate_metadata(load_cached_data=True)

    # 2. Feature Extraction
    # Extracts features from BSON to .npy files if not already cached.
    # We use the full dataset (debug=False) to ensure high accuracy.
    print("Step 2: Running Feature Extraction...")
    run_feature_extraction(load_cached_data=True, debug=False)

    # 3. Train Ensemble
    # Trains 5 MLPs on the cached features with MixUp regularization.
    print("Step 3: Training Ensemble...")
    models = train_ensemble()

    # 4. Validation
    print("Step 4: Validating...")
    if not os.path.exists(VAL_FEATURES) or not os.path.exists(VAL_LABELS):
        raise FileNotFoundError("Validation features not found.")

    val_features = np.load(VAL_FEATURES)
    val_labels = np.load(VAL_LABELS)

    val_dataset = FeatureDataset(val_features, val_labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    # Inference on Validation Set
    all_preds = []
    # Set all models to eval mode
    for model in models:
        model.eval()

    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(DEVICE)
            # Average predictions across the ensemble
            avg_probs = torch.zeros((inputs.size(0), NUM_CLASSES), device=DEVICE)
            for model in models:
                outputs = model(inputs)
                avg_probs += torch.softmax(outputs, dim=1)
            avg_probs /= len(models)
            all_preds.append(avg_probs.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds)

    # Calculate Metric
    acc = np.mean(y_pred == val_labels)
    # Print full precision as required
    print(f"Final Validation Metric: {acc}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    errors = (y_pred != val_labels).astype(int)
    # Calculate L2 norm of features as a proxy for signal magnitude/complexity
    feat_norms = np.linalg.norm(val_features, axis=1)
    # Calculate correlation between Error and Feature Magnitude
    correlation = np.corrcoef(errors, feat_norms)[0, 1]
    print(f"Correlation between Error and Feature Magnitude: {correlation:.4f}")

    # 5. Submission
    THRESHOLD = 0.50636
    if acc > THRESHOLD:
        print(f"Validation metric {acc} > {THRESHOLD}. Generating submission...")

        if not os.path.exists(TEST_FEATURES) or not os.path.exists(TEST_IDS):
            raise FileNotFoundError("Test features not found.")

        test_features = np.load(TEST_FEATURES)
        test_ids = np.load(TEST_IDS)

        test_dataset = FeatureDataset(test_features)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == "cuda"),
        )

        test_pred_indices = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(DEVICE)
                avg_probs = torch.zeros((inputs.size(0), NUM_CLASSES), device=DEVICE)
                for model in models:
                    outputs = model(inputs)
                    avg_probs += torch.softmax(outputs, dim=1)
                avg_probs /= len(models)
                test_pred_indices.append(avg_probs.argmax(dim=1).cpu().numpy())

        test_pred_indices = np.concatenate(test_pred_indices)

        # Decode category indices to original category_ids
        encoder = get_category_encoder(load_cached_data=True)
        test_pred_cats = encoder.inverse_transform(test_pred_indices)

        submission = pd.DataFrame({"_id": test_ids, "category_id": test_pred_cats})

        submission_path = "./submission/submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"Validation metric {acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
