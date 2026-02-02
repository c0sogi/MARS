import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Import from provided library
from library.config import set_seed, EXPERT_LIBRARY, SUBMISSION_PATH
from library.data_loader import load_datasets, get_feature_columns
from library.feature_engineering import get_morph_poly_features
from library.expert_library import build_expert_library
from library.ensemble_selection import GreedyEnsembleSelector


def run_pipeline():
    # 1. Setup
    print("Initializing environment...")
    set_seed(42)

    # 2. Load Data
    print("Loading datasets...")
    # load_cached_data=True allows using parquet cache if available for speed
    df_train, df_val, df_test = load_datasets(load_cached_data=True)

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    # 3. Encode Labels
    # We use a single encoder to ensure consistency across all steps
    print("Encoding target labels...")
    le = LabelEncoder()
    # Fit on training species to establish the class mapping
    le.fit(df_train["species"])

    y_train = le.transform(df_train["species"])
    y_val = le.transform(df_val["species"])

    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # 4. Prepare Feature Views

    # View A: Global (Provided Tabular Features)
    print("Preparing 'global' feature view...")
    global_cols = get_feature_columns(df_train)

    X_global_train = df_train[global_cols].values
    X_global_val = df_val[global_cols].values
    X_global_test = df_test[global_cols].values

    # View B: Morphometric + Polynomial (Computed from Images)
    print(
        "Preparing 'morph_poly' feature view (image processing + polynomial expansion)..."
    )

    # Helper to extract features and ensure alignment with the main dataframe
    def get_X_morph(df, name):
        # This function extracts physical features from images and applies polynomial expansion
        df_poly = get_morph_poly_features(df, dataset_name=name, load_cached_data=True)

        # Ensure alignment by merging on ID.
        # Although processing usually preserves order, merging is safer.
        if "id" in df_poly.columns:
            df_merged = pd.merge(df[["id"]], df_poly, on="id", how="left")
            # Drop ID to get the feature matrix
            return df_merged.drop(columns=["id"]).values
        return df_poly.values

    X_morph_train = get_X_morph(df_train, "train")
    X_morph_val = get_X_morph(df_val, "val")
    X_morph_test = get_X_morph(df_test, "test")

    print(f"Global features shape: {X_global_train.shape}")
    print(f"Morph-Poly features shape: {X_morph_train.shape}")

    # Store views in a dict for easy access by the expert configuration
    feature_views = {
        "global": (X_global_train, X_global_val, X_global_test),
        "morph_poly": (X_morph_train, X_morph_val, X_morph_test),
    }

    # 5. Build and Train Experts
    print("Building and training experts...")
    experts = build_expert_library()

    # Dictionaries to store predictions for the ensemble selector
    val_preds_dict = {}
    test_preds_dict = {}

    for config in EXPERT_LIBRARY:
        expert_id = config["id"]
        view_name = config["feature_view"]

        print(f"Training Expert: {expert_id} (View: {view_name})")

        # Retrieve the correct feature set for this expert
        X_train_curr, X_val_curr, X_test_curr = feature_views[view_name]

        # Retrieve the pipeline
        pipeline = experts[expert_id]

        # Fit the pipeline
        pipeline.fit(X_train_curr, y_train)

        # Predict probabilities
        val_probs = pipeline.predict_proba(X_val_curr)
        test_probs = pipeline.predict_proba(X_test_curr)

        # Store predictions
        val_preds_dict[expert_id] = val_probs
        test_preds_dict[expert_id] = test_probs

        # Quick Validation Score for individual expert
        loss = log_loss(y_val, val_probs)
        print(f"  -> Val LogLoss: {loss:.4f}")

    # 6. Ensemble Selection
    print("Optimizing ensemble weights...")
    # Initialize selector with a moderate number of iterations for the demo
    selector = GreedyEnsembleSelector(n_iterations=20)

    # Fit the selector using validation predictions and true labels
    selector.fit(val_preds_dict, y_val)

    # 7. Generate Final Predictions
    print("Generating submission...")
    # Compute weighted average of test predictions
    final_test_probs = selector.predict(test_preds_dict)

    # 8. Create Submission File
    # The submission requires columns to be the species names
    submission_df = pd.DataFrame(final_test_probs, columns=le.classes_)

    # Insert the ID column at the beginning
    submission_df.insert(0, "id", df_test["id"])

    # Save to the designated submission path
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    # 9. Final Validation
    print("Validating submission format...")
    saved_df = pd.read_csv(SUBMISSION_PATH)

    # Check 1: Row count matches test set
    assert len(saved_df) == len(df_test), "Submission row count mismatch"

    # Check 2: Column count (ID + 99 species)
    assert saved_df.shape[1] == 100, f"Expected 100 columns, got {saved_df.shape[1]}"

    # Check 3: Probabilities range [0, 1]
    probs_only = saved_df.drop(columns=["id"]).values
    assert np.all(probs_only >= 0) and np.all(
        probs_only <= 1.0 + 1e-6
    ), "Probabilities out of range"

    # Check 4: Probabilities sum to 1 (approx)
    row_sums = probs_only.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("Demo completed successfully!")


if __name__ == "__main__":
    run_pipeline()
