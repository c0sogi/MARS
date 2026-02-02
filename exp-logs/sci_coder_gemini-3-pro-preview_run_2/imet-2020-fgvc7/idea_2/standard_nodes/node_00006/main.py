import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# Import provided library modules
from library.config import Config
from library.train import run_training
from library.optimize import (
    get_validation_predictions,
    optimize_thresholds,
    evaluate_with_thresholds,
)
from library.dataset import get_test_loader
from library.utils import seed_everything, get_device
from library.model import ArtworkConvNeXt


def main():
    # --- 1. Configuration for Fast Baseline ---
    # We use a subset of training data (50k) but the full validation set (24k < 50k)
    # to ensure speed while maintaining a rigorous evaluation.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50000
    Config.NUM_EPOCHS = 5
    Config.BATCH_SIZE = 64  # Increase batch size for A100

    print(
        f"Configuration set: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, "
        f"Debug={Config.DEBUG}, Sample Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # --- 2. Training ---
    print("\n=== Starting Training ===")
    # run_training handles seeding and the training loop
    run_training(debug=Config.DEBUG)

    # --- 3. Validation and Threshold Optimization ---
    print("\n=== Starting Validation & Threshold Optimization ===")
    # Force reload of data to ensure we use the model we just trained
    val_logits, val_targets = get_validation_predictions(load_cached_data=False)

    # Optimize thresholds per class
    best_thresholds = optimize_thresholds(val_logits, val_targets)

    # Calculate and print Final Validation Metric
    final_f1 = evaluate_with_thresholds(val_logits, val_targets, best_thresholds)
    print(f"Final Validation Metric: {final_f1}")

    # --- 4. Failure Analysis ---
    print("\n=== Starting Failure Analysis ===")
    perform_failure_analysis(val_logits, val_targets)

    # --- 5. Submission ---
    print("\n=== Checking Submission Criteria ===")
    TARGET_SCORE = 0.5559147184169592

    if final_f1 > TARGET_SCORE:
        print(f"Score {final_f1} > {TARGET_SCORE}. Generating submission...")
        generate_submission(best_thresholds)
    else:
        print(f"Score {final_f1} <= {TARGET_SCORE}. Submission skipped.")


def perform_failure_analysis(val_logits, val_targets):
    """
    Analyzes model errors against metadata features.
    """
    # Calculate error magnitude per sample (Mean Absolute Error)
    val_probs = 1 / (1 + np.exp(-val_logits))
    # MAE: average absolute difference between prediction probability and target (0 or 1)
    sample_mae = np.mean(np.abs(val_probs - val_targets), axis=1)

    # Load validation metadata
    # Note: get_validation_predictions loads data sequentially without shuffling (shuffle=False in loader)
    # So we can align directly with the metadata CSV if we load it similarly.
    # However, get_dataloaders might apply the debug head() slice.
    df_val = pd.read_csv(Config.VAL_META_PATH)
    if Config.DEBUG:
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)

    # Add error column
    df_val["error_magnitude"] = sample_mae

    # Feature 1: Label Count
    df_val["label_count"] = df_val["attribute_ids"].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) and x != "" else 0
    )

    # Feature 2: File Size (proxy for image complexity/quality)
    # We compute this on the fly
    def get_file_size(rel_path):
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        if os.path.exists(full_path):
            return os.path.getsize(full_path)
        return 0

    df_val["file_size"] = df_val["file_path"].apply(get_file_size)

    # Calculate Correlations
    correlations = df_val[["error_magnitude", "label_count", "file_size"]].corr()[
        "error_magnitude"
    ]

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.drop("error_magnitude"))


def generate_submission(thresholds):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = get_device()

    # Load Test Loader
    test_loader = get_test_loader(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Load Model
    model = ArtworkConvNeXt(num_classes=Config.NUM_CLASSES, pretrained=False)
    # Load the best model saved during training
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    submission_ids = []
    submission_attrs = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Apply thresholds
            # thresholds shape: (C,) -> (1, C)
            # probs shape: (B, C)
            preds_binary = probs >= thresholds[None, :]

            # Convert to string format
            for i in range(len(ids)):
                # Get indices where prediction is True
                true_indices = np.where(preds_binary[i])[0]
                # Join with spaces
                attr_str = " ".join(map(str, true_indices))

                submission_ids.append(ids[i])
                submission_attrs.append(attr_str)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": submission_ids, "attribute_ids": submission_attrs})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)
    main()
