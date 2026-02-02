import sys
import os
import numpy as np
import pandas as pd
import tqdm

# 1. Patch tqdm to suppress progress bars before importing library modules that use it
tqdm.tqdm = lambda x, *args, **kwargs: x

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_log_loss,
    save_submission,
    get_logger,
)
from library.data_processor import LeafDataManager
from library.model_factory import create_classifier

# 2. Configuration Override for Fast Demonstration
# We modify the Config class directly to run in a lightweight debug mode
print("Configuring environment for fast demonstration...")
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = (
    24  # Small multiple of 3 (centroids) and manageable for batching
)
Config.N_FOLDS = 2  # Minimal folds for demo
Config.PCA_VARIANCE = (
    0.95  # Slightly lower variance for faster PCA convergence on small data
)

# Setup Logger
logger = get_logger(name="demo_script")


def main():
    seed_everything(Config.SEED)

    # ==========================================
    # 3. Data Loading & Processing
    # ==========================================
    print("\n--- Step 1: Data Loading & Manifold Densification ---")
    data_manager = LeafDataManager()

    # Load Training Data (Densified)
    # This triggers FeatureExtractor -> DINOv2/ConvNeXt -> Centroid Aggregation
    train_data = data_manager.get_dataset(stage="train", load_cached_data=False)

    X_train_full = train_data["X"]
    y_train_full = train_data["y"]
    ids_train_full = train_data["ids"]

    # Verification of Densification Logic
    # We expect the number of samples to be 3 * DEBUG_SAMPLE_SIZE (or close to it if filtered)
    # The feature dimension should be 1024 (DINO) + 1536 (Conv) + 192 (Tabular) = 2752
    expected_dim = 1024 + 1536 + 192
    print(f"Train Data Shape: {X_train_full.shape}")
    print(f"Train Labels Shape: {y_train_full.shape}")

    if X_train_full.shape[1] != expected_dim:
        raise AssertionError(
            f"Feature dimension mismatch. Expected {expected_dim}, got {X_train_full.shape[1]}"
        )

    if len(X_train_full) != len(y_train_full):
        raise AssertionError("Mismatch between features and labels count.")

    # Verify Manifold Densification factor (should be 3x unique IDs)
    unique_ids_count = len(np.unique(ids_train_full))
    if len(X_train_full) != unique_ids_count * 3:
        raise AssertionError(
            f"Densification failed. Expected {unique_ids_count * 3} samples, got {len(X_train_full)}"
        )

    print("Data loaded and verified successfully.")

    # ==========================================
    # 4. Model Pipeline Construction
    # ==========================================
    print("\n--- Step 2: Model Pipeline Construction ---")
    pipeline = create_classifier()
    print("Pipeline created successfully.")
    print(pipeline)

    # ==========================================
    # 5. Cross-Validation Demonstration
    # ==========================================
    print("\n--- Step 3: Cross-Validation Loop ---")

    # Get stratified folds
    # Note: We use the manager's method to ensure we don't split centroids of the same image
    folds = data_manager.get_stratified_folds(
        X_train_full, y_train_full, ids_train_full, n_folds=Config.N_FOLDS
    )

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"Running Fold {fold_idx + 1}/{Config.N_FOLDS}...")

        # Split data
        X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        # Verify disjoint split
        assert (
            len(set(train_idx).intersection(set(val_idx))) == 0
        ), "Train and Validation indices overlap!"

        # Fit Pipeline
        # LDA handles string class labels automatically
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred_prob = pipeline.predict_proba(X_val)

        # Filter validation samples to only those with classes seen during training
        # This handles disjoint label sets caused by aggressive subsampling (Cite debug_lesson_1)
        valid_mask = np.isin(y_val, pipeline.classes_)

        if valid_mask.sum() > 0:
            y_val_filtered = y_val[valid_mask]
            y_pred_filtered = y_pred_prob[valid_mask]

            # Calculate Metric
            # We pass pipeline.classes_ to ensure correct mapping
            loss = calculate_log_loss(
                y_val_filtered, y_pred_filtered, labels=pipeline.classes_
            )
            fold_scores.append(loss)

            print(f"Fold {fold_idx + 1} Log Loss: {loss:.4f}")
        else:
            print(
                f"Fold {fold_idx + 1}: No common classes between train and val. Skipping metric."
            )

    avg_loss = np.mean(fold_scores)
    print(f"Average CV Log Loss: {avg_loss:.4f}")

    # ==========================================
    # 6. Full Training & Inference
    # ==========================================
    print("\n--- Step 4: Full Training & Inference ---")

    # Refit on all training data
    pipeline.fit(X_train_full, y_train_full)

    # Load Test Data
    test_data = data_manager.get_dataset(stage="test", load_cached_data=False)
    X_test = test_data["X"]
    ids_test = test_data["ids"]

    print(f"Test Data Shape: {X_test.shape}")

    # Predict on Test Data
    # Note: Test data is also densified (3 centroids per image).
    # For submission, we need to aggregate predictions back to 1 per image.
    # A common strategy is averaging probabilities across the 3 centroids.
    probs_densified = pipeline.predict_proba(X_test)

    # Aggregate predictions: Average every 3 rows
    # Reshape to (N_images, 3, N_classes)
    n_test_images = len(np.unique(ids_test))
    n_classes = len(pipeline.classes_)

    # Ensure divisible by 3
    assert len(probs_densified) == n_test_images * 3

    probs_reshaped = probs_densified.reshape(n_test_images, 3, n_classes)
    probs_final = np.mean(probs_reshaped, axis=1)

    # Get unique IDs (preserving order, assuming sorted blocks of 3)
    # ids_test is like [ID1, ID1, ID1, ID2, ID2, ID2...]
    ids_final = ids_test[::3]

    print(f"Aggregated Predictions Shape: {probs_final.shape}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    print("\n--- Step 5: Submission Generation ---")

    submission_path = "./submission_demo.csv"

    # Use the classes from the fitted pipeline for column headers
    class_names = list(pipeline.classes_)

    save_submission(ids_final, class_names, probs_final, filename=submission_path)

    # ==========================================
    # 8. Final Verification
    # ==========================================
    print("\n--- Step 6: Verification ---")

    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")
    print("Head:")
    print(df_sub.head(3))

    # Check columns
    expected_cols = ["id"] + class_names
    if list(df_sub.columns) != expected_cols:
        raise ValueError("Submission columns do not match expected classes.")

    # Check ID integrity
    if not np.array_equal(df_sub["id"].values, ids_final):
        raise ValueError("Submission IDs do not match test IDs.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
