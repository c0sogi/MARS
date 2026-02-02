import sys
import os
import numpy as np
import pandas as pd
import warnings
from library.config import SUBMISSION_PATH, NUM_CLASSES
from library.utils import set_seed, save_submission
from library.data_loader import create_dataloaders
from library.feature_extractor import FeatureExtractor
from library.classifier import LogRegClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def align_predictions(raw_preds, model_classes, total_classes):
    """
    Aligns predictions from a model trained on a subset of classes
    to the full set of classes required for submission.

    Args:
        raw_preds (np.ndarray): Predictions of shape (N, n_trained_classes).
        model_classes (np.ndarray): Indices of classes the model was trained on.
        total_classes (int): Total number of expected classes (120).

    Returns:
        np.ndarray: Aligned predictions of shape (N, total_classes).
    """
    if raw_preds.shape[1] == total_classes:
        return raw_preds

    aligned = np.zeros((raw_preds.shape[0], total_classes), dtype=raw_preds.dtype)

    # Map columns from the raw prediction to the correct index in the full matrix
    for i, class_idx in enumerate(model_classes):
        aligned[:, int(class_idx)] = raw_preds[:, i]

    return aligned


def main():
    print("Starting End-to-End Pipeline Demonstration...")
    set_seed(42)

    # ==========================================
    # 1. Data Loading
    # ==========================================
    # We use a debug_limit of 500 to ensure the code runs very fast for demonstration.
    print("\n[1/5] Loading Data (Subset)...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        debug_limit=500, batch_size=32, num_workers=2
    )

    # Validation: Ensure we have the correct class definitions
    assert (
        len(class_names) == NUM_CLASSES
    ), f"Expected {NUM_CLASSES} class names, got {len(class_names)}"
    print(f"DataLoaders created. Total defined classes: {len(class_names)}")

    # ==========================================
    # 2. Feature Extraction
    # ==========================================
    print("\n[2/5] Extracting Features...")
    extractor = FeatureExtractor()

    # Force re-computation (load_cached_data=False) because we are using a subset
    # and want to avoid loading cached features from a previous full run.
    X_train, y_train = extractor.get_train_features(
        train_loader, load_cached_data=False
    )
    X_val, y_val = extractor.get_val_features(val_loader, load_cached_data=False)
    X_test, test_ids = extractor.get_test_features(test_loader, load_cached_data=False)

    # Validation: Check feature dimensions (ResNet50 output is 2048)
    assert X_train.shape[1] == 2048, f"Expected 2048 features, got {X_train.shape[1]}"
    assert len(X_train) == len(y_train), "Mismatch between training features and labels"
    print(f"Extracted training features shape: {X_train.shape}")

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("\n[3/5] Training Classifier...")
    classifier = LogRegClassifier()

    # Train the model (force retraining on this subset)
    classifier.train(X_train, y_train, load_cached_model=False)

    trained_classes = classifier.get_classes()
    print(
        f"Model trained on {len(trained_classes)} unique classes found in the subset."
    )

    # ==========================================
    # 4. Evaluation
    # ==========================================
    print("\n[4/5] Evaluating Model...")

    # Robustness Check: Since we are using a random subset, the validation set might
    # contain classes that were not in the training subset.
    # We filter the validation set to only evaluate on classes the model knows.
    known_class_mask = np.isin(y_val, trained_classes)

    if known_class_mask.sum() > 0:
        X_val_filtered = X_val[known_class_mask]
        y_val_filtered = y_val[known_class_mask]

        loss = classifier.evaluate(X_val_filtered, y_val_filtered)
        print(f"Validation Log Loss (on known classes): {loss:.4f}")
    else:
        print(
            "Warning: No overlap between training classes and validation subset. Skipping evaluation."
        )

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[5/5] Generating Submission...")

    # Get raw probabilities
    raw_preds = classifier.predict(X_test)

    # Align predictions to the full 120-column format
    # (Fills missing columns with 0.0)
    final_preds = align_predictions(raw_preds, trained_classes, NUM_CLASSES)

    # Validation: Check output shape
    assert final_preds.shape == (
        len(test_ids),
        NUM_CLASSES,
    ), f"Final prediction shape mismatch. Expected {(len(test_ids), NUM_CLASSES)}, got {final_preds.shape}"

    # Save to disk
    save_submission(final_preds, test_ids, class_names, SUBMISSION_PATH)

    # Final Verification
    if os.path.exists(SUBMISSION_PATH):
        print(f"Submission successfully saved to: {SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Failed to generate submission file.")


if __name__ == "__main__":
    main()
