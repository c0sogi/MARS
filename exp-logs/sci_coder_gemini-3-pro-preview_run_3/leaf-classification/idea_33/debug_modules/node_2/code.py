import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided library components
from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.feature_extractor import FeatureExtractor
from library.manifold_processor import ManifoldDensifier
from library.topology_pipeline import build_model_pipeline


def main():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("--- Setting up environment ---")
    seed_everything(Config.SEED)

    # Override Config for speed (Demonstration Mode)
    # We use a small subset of data to ensure the script runs quickly.
    Config.DEBUG = True
    Config.DEBUG_LIMIT = (
        12  # Small number, divisible by batch size ideally, but robust to any
    )

    # Ensure output directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Limit: {Config.DEBUG_LIMIT}")

    # ==========================================
    # 2. Feature Extraction
    # ==========================================
    print("\n--- Step 1: Feature Extraction ---")
    extractor = FeatureExtractor()

    # Extract features for all splits.
    # We force extraction (load_cached_data=False) to demonstrate the logic.
    train_data, val_data, test_data = extractor.extract_all(load_cached_data=False)

    # Unpack data
    dino_train, conv_train, ids_train = train_data
    dino_val, conv_val, ids_val = val_data
    dino_test, conv_test, ids_test = test_data

    # Verification
    print(f"Train DINO shape: {dino_train.shape}")
    print(f"Train Conv shape: {conv_train.shape}")

    # Assertions
    # Shapes should be (N, 12, Embed_Dim)
    assert (
        len(dino_train.shape) == 3 and dino_train.shape[1] == 12
    ), "DINO features must have 12 views"
    assert (
        len(conv_train.shape) == 3 and conv_train.shape[1] == 12
    ), "Conv features must have 12 views"
    assert (
        len(ids_train) == dino_train.shape[0]
    ), "Mismatch between IDs and feature count"

    if Config.DEBUG:
        assert len(ids_train) <= Config.DEBUG_LIMIT, "Debug limit not respected"

    # ==========================================
    # 3. Manifold Densification
    # ==========================================
    print("\n--- Step 2: Manifold Densification ---")
    densifier = ManifoldDensifier()

    # Process Train
    # Returns flattened arrays where N images become 3*N samples (3 centroids per image)
    (
        dino_dense_train,
        conv_dense_train,
        tab_dense_train,
        ids_dense_train,
        y_dense_train,
    ) = densifier.process_split(
        "train", dino_train, conv_train, ids_train, load_cached_data=False
    )

    # Process Val
    (dino_dense_val, conv_dense_val, tab_dense_val, ids_dense_val, y_dense_val) = (
        densifier.process_split(
            "val", dino_val, conv_val, ids_val, load_cached_data=False
        )
    )

    # Process Test (y_dense will be None)
    (dino_dense_test, conv_dense_test, tab_dense_test, ids_dense_test, _) = (
        densifier.process_split(
            "test", dino_test, conv_test, ids_test, load_cached_data=False
        )
    )

    # Verification
    print(f"Densified Train shape: {dino_dense_train.shape}")

    # Assertions
    # The densifier collapses 12 views into 3 centroids, so samples should triple
    assert (
        dino_dense_train.shape[0] == dino_train.shape[0] * 3
    ), "Densification did not triple the sample count"
    assert tab_dense_train.shape[1] == 192, "Tabular features should have 192 columns"
    assert y_dense_train is not None, "Training labels missing"

    # ==========================================
    # 4. Data Preparation for Pipeline
    # ==========================================
    print("\n--- Step 3: Data Preparation ---")

    # Concatenate features: [DINO, ConvNeXt, Tabular]
    def prepare_X(dino, conv, tab):
        return np.hstack([dino, conv, tab])

    X_train = prepare_X(dino_dense_train, conv_dense_train, tab_dense_train)
    X_val = prepare_X(dino_dense_val, conv_dense_val, tab_dense_val)
    X_test = prepare_X(dino_dense_test, conv_dense_test, tab_dense_test)

    # Dimensions for the pipeline
    dino_dim = dino_dense_train.shape[1]
    conv_dim = conv_dense_train.shape[1]
    tab_dim = tab_dense_train.shape[1]

    print(f"Combined Feature Vector Length: {X_train.shape[1]}")

    # ==========================================
    # 5. Model Training
    # ==========================================
    print("\n--- Step 4: Model Training ---")

    pipeline = build_model_pipeline(dino_dim, conv_dim, tab_dim)

    print("Fitting pipeline (PCA + QuantileTransformer + LDA)...")
    pipeline.fit(X_train, y_dense_train)

    # ==========================================
    # 6. Validation & Evaluation
    # ==========================================
    print("\n--- Step 5: Validation ---")

    # Predict probabilities on densified validation set (3 samples per image)
    probs_dense_val = pipeline.predict_proba(X_val)

    # Aggregate predictions: Average probabilities across the 3 centroids for each image
    # Create a DataFrame to facilitate grouping
    val_pred_df = pd.DataFrame(probs_dense_val, columns=pipeline.classes_)
    val_pred_df["id"] = ids_dense_val

    # Group by ID and take mean
    val_agg_df = val_pred_df.groupby("id").mean()

    # Align ground truth labels
    # We need the original label for each ID. Since ids_dense_val and y_dense_val are aligned:
    val_gt_df = pd.DataFrame({"id": ids_dense_val, "species": y_dense_val})
    val_gt_df = val_gt_df.drop_duplicates(subset=["id"]).set_index("id")

    # Ensure indices match
    common_ids = val_agg_df.index.intersection(val_gt_df.index)
    val_agg_df = val_agg_df.loc[common_ids]
    val_gt_df = val_gt_df.loc[common_ids]

    # Calculate Log Loss
    # Clip probabilities first
    y_true = val_gt_df["species"].values
    y_pred = clip_probabilities(val_agg_df.values)

    try:
        score = log_loss(y_true, y_pred, labels=pipeline.classes_)
        print(f"Validation Log Loss: {score:.4f}")
    except ValueError as e:
        # This might happen in debug mode if not all classes are present in the small subset
        print(
            f"Validation Log Loss could not be calculated (likely due to missing classes in debug subset): {e}"
        )

    # ==========================================
    # 7. Inference & Submission
    # ==========================================
    print("\n--- Step 6: Inference & Submission ---")

    # Predict on test set
    probs_dense_test = pipeline.predict_proba(X_test)

    # Aggregate test predictions
    test_pred_df = pd.DataFrame(probs_dense_test, columns=pipeline.classes_)
    test_pred_df["id"] = ids_dense_test
    test_agg_df = test_pred_df.groupby("id").mean().reset_index()

    # Load sample submission to get correct column order
    sample_sub = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))

    # Prepare final submission dataframe
    submission = pd.DataFrame()
    submission["id"] = test_agg_df["id"]

    # Fill columns
    for col in sample_sub.columns:
        if col == "id":
            continue
        if col in test_agg_df.columns:
            submission[col] = test_agg_df[col]
        else:
            # If a class was not seen in training (possible in debug mode), fill with 0
            submission[col] = 0.0

    # Clip probabilities
    feature_cols = [c for c in submission.columns if c != "id"]
    submission[feature_cols] = clip_probabilities(submission[feature_cols].values)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")

    # Final Verification
    assert not submission.isnull().values.any(), "Submission contains NaNs"
    assert (
        submission.shape[1] == sample_sub.shape[1]
    ), "Submission column count mismatch"

    print("\n--- Execution Complete ---")


if __name__ == "__main__":
    main()
