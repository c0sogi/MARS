import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import components from the provided library
from library.utils import set_seed, setup_logger
from library.data_loader import get_data_splits
from library.feature_extractor import EmbeddingGenerator, extract_metadata_features
from library.pipeline_builder import (
    build_adrsf_pipeline,
    combine_features,
    DIM_HIGH_RES,
    DIM_LOW_RES,
    DIM_METADATA,
)


def run_demo():
    print(">>> Starting Library Demo Execution...")

    # 1. Setup & Reproducibility
    set_seed(42)

    # 2. Data Loading
    # We load the full splits, but we will slice them immediately for the demo
    print("\n[Step 1] Loading Data...")
    train_df, val_df, test_df = get_data_splits(load_cached_data=True)

    print(f"    Full Train Shape: {train_df.shape}")
    print(f"    Full Val Shape:   {val_df.shape}")
    print(f"    Full Test Shape:  {test_df.shape}")

    # 3. Create Subsets for Speed
    # To ensure this script completes quickly, we use a small subset (e.g., 50 samples)
    SUBSET_SIZE = 50
    print(
        f"\n[Step 2] Creating subsets of size {SUBSET_SIZE} for rapid demonstration..."
    )

    train_subset = train_df.iloc[:SUBSET_SIZE].copy()
    val_subset = val_df.iloc[:SUBSET_SIZE].copy()
    test_subset = test_df.iloc[:SUBSET_SIZE].copy()

    # 4. Feature Extraction
    print("\n[Step 3] Extracting Features...")
    emb_gen = EmbeddingGenerator()

    # Helper function to process a dataframe into the final feature matrix
    def get_features(df, split_name):
        # Generate Embeddings
        # We use unique split names (e.g., 'demo_train') to create separate cache files
        # and avoid overwriting the main experiment's cache.
        emb_high, emb_low = emb_gen.process_split(
            df,
            split_name=f"demo_{split_name}",
            load_cached_data=False,  # Force generation for demo purposes
            batch_size=16,
        )

        # Extract Metadata
        meta = extract_metadata_features(df)

        # Combine
        X = combine_features(emb_high, emb_low, meta)
        return X

    # Process Train
    print("    Processing Train Subset...")
    X_train = get_features(train_subset, "train")
    y_train = train_subset["requester_received_pizza"].values.astype(int)

    # Process Validation
    print("    Processing Validation Subset...")
    X_val = get_features(val_subset, "val")
    y_val = val_subset["requester_received_pizza"].values.astype(int)

    # Validation Logic: Check Shapes
    expected_dim = DIM_HIGH_RES + DIM_LOW_RES + DIM_METADATA
    assert X_train.shape == (
        SUBSET_SIZE,
        expected_dim,
    ), f"Expected shape ({SUBSET_SIZE}, {expected_dim}), got {X_train.shape}"
    assert X_val.shape == (SUBSET_SIZE, expected_dim)
    print(f"    Feature Matrix Shape Verified: {X_train.shape}")

    # 5. Model Building & Training
    print("\n[Step 4] Building and Training Pipeline...")
    # We reduce n_estimators and pca_components for speed in this demo
    pipeline = build_adrsf_pipeline(
        pca_components=16,
        n_estimators=5,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    print("    Training complete.")

    # 6. Evaluation
    print("\n[Step 5] Evaluating on Validation Subset...")
    val_preds = pipeline.predict_proba(X_val)[:, 1]

    # Calculate metric if possible (requires both classes to be present in subset)
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, val_preds)
        print(f"    Validation AUC (Subset): {auc:.4f}")
    else:
        print("    Skipping AUC calculation (only one class present in subset).")

    # 7. Inference
    print("\n[Step 6] Running Inference on Test Subset...")
    X_test = get_features(test_subset, "test")
    test_preds = pipeline.predict_proba(X_test)[:, 1]

    # 8. Submission Generation
    print("\n[Step 7] Generating Submission File...")
    submission_dir = "./demo_submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df = pd.DataFrame(
        {
            "request_id": test_subset["request_id"],
            "requester_received_pizza": test_preds,
        }
    )

    submission_df.to_csv(submission_path, index=False)
    print(f"    Submission saved to: {submission_path}")

    # Final Verification
    saved_df = pd.read_csv(submission_path)
    assert saved_df.shape == (SUBSET_SIZE, 2), "Submission shape mismatch"
    assert "request_id" in saved_df.columns
    assert "requester_received_pizza" in saved_df.columns
    assert (
        saved_df["requester_received_pizza"].between(0, 1).all()
    ), "Probabilities out of range"

    print("\n>>> Demo Execution Successfully Completed.")


if __name__ == "__main__":
    run_demo()
