import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

# Ensure reproducible results
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library import config, utils, data_loader, transformers, expert_pipelines, ensemble


def main():
    print("=== Starting Library Verification and Demonstration ===\n")

    # =========================================================================
    # 1. Data Loading and Feature Subsets
    # =========================================================================
    print("--- 1. Loading Datasets ---")
    # Load datasets using the data_loader module
    # This handles metadata loading, morphometric extraction (or caching), and merging
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        test_ids,
        classes,
        feature_subsets,
    ) = data_loader.load_datasets(load_cached_data=True)

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")
    print(f"Number of Classes: {len(classes)}")

    # Verify Feature Subsets
    assert "global" in feature_subsets
    assert "margin" in feature_subsets
    assert "shape" in feature_subsets
    assert "texture" in feature_subsets
    assert "morphometrics" in feature_subsets
    print("Feature subsets verified successfully.")

    # =========================================================================
    # 2. Testing Custom Transformers
    # =========================================================================
    print("\n--- 2. Testing Custom Transformers ---")

    # Test Float64Transformer
    f64_transformer = transformers.Float64Transformer()
    dummy_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    transformed_data = f64_transformer.transform(dummy_data)
    assert transformed_data.dtype == np.float64
    print("Float64Transformer: type casting verified.")

    # Test LDADimensionalityReducer
    # We use a small subset for speed
    subset_idx = np.random.choice(len(X_train), 100, replace=False)
    X_sub = X_train.iloc[subset_idx]
    y_sub = y_train[subset_idx]

    n_components = 5
    lda_reducer = transformers.LDADimensionalityReducer(n_components=n_components)
    lda_reducer.fit(X_sub, y_sub)
    X_lda = lda_reducer.transform(X_sub)

    assert X_lda.shape == (100, n_components)
    assert X_lda.dtype == np.float64
    print(f"LDADimensionalityReducer: output shape {X_lda.shape} verified.")

    # =========================================================================
    # 3. Testing Expert Pipeline Builders
    # =========================================================================
    print("\n--- 3. Testing Expert Pipeline Builders ---")

    # Define a small helper to test a pipeline
    def test_pipeline(name, pipeline):
        print(f"Testing {name}...")
        # Fit on small subset
        pipeline.fit(X_sub, y_sub)
        # Transform
        res = pipeline.transform(X_sub)
        # Check basic properties
        assert isinstance(res, np.ndarray) or hasattr(res, "toarray")
        assert res.shape[0] == len(X_sub)
        # Check for NaNs
        if hasattr(res, "toarray"):
            res = res.toarray()
        assert not np.isnan(res).any()
        print(f"  -> {name} passed. Output shape: {res.shape}")

    # A. Global Pipeline
    pipe_global = expert_pipelines.build_global_pipeline("Marginal", feature_subsets)
    test_pipeline("Global Pipeline (Marginal)", pipe_global)

    # B. Stratified Rotational Pipeline
    pipe_strat = expert_pipelines.build_stratified_rotational_pipeline(feature_subsets)
    test_pipeline("Stratified Rotational Pipeline", pipe_strat)

    # C. Intra-Domain Pipeline
    pipe_intra = expert_pipelines.build_intra_domain_pipeline(feature_subsets)
    test_pipeline("Intra-Domain Pipeline", pipe_intra)

    # D. Inter-Domain Pipeline
    pair = ("margin", "shape")
    pipe_inter = expert_pipelines.build_inter_domain_pipeline(pair, feature_subsets)
    test_pipeline(f"Inter-Domain Pipeline {pair}", pipe_inter)

    # E. Morphometric Pipeline
    # Note: If morphometrics failed to extract (all zeros), this might throw singular matrix errors in PowerTransformer
    # dependent on data distribution. Assuming data_loader worked, this should be fine.
    pipe_morph = expert_pipelines.build_morphometric_pipeline(feature_subsets)
    test_pipeline("Morphometric Pipeline", pipe_morph)

    # =========================================================================
    # 4. Testing Ensemble (HDME)
    # =========================================================================
    print("\n--- 4. Testing HDME Ensemble ---")

    # Modify config for speed: Reduce LDA shrinkage candidates
    original_shrinkage = config.LDA_SHRINKAGE_CANDIDATES
    config.LDA_SHRINKAGE_CANDIDATES = [0.5]  # Use single value for demo speed

    # Initialize Ensemble with small max size for Greedy Selection
    hdme = ensemble.HDME_Ensemble(max_ensemble_size=3)

    print(
        "Fitting Ensemble (this involves generating candidates and greedy selection)..."
    )
    # We use the full training set here to ensure LDA has enough samples for classes
    hdme.fit(X_train, y_train, X_val, y_val, feature_subsets)

    print("Selected Experts Configuration:")
    for conf in hdme.selected_config:
        print(f"  - {conf}")

    assert len(hdme.selected_config) > 0, "Ensemble failed to select any experts."

    # =========================================================================
    # 5. Prediction and Scoring
    # =========================================================================
    print("\n--- 5. Prediction and Validation ---")

    # Combine Train and Val for final retraining (simulated by the predict method)
    X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train, y_val], axis=0)

    # Predict on Test Set
    print("Predicting on Test set...")
    y_pred_test = hdme.predict(X_full, y_full, X_test, feature_subsets)

    # Verify Prediction Properties
    assert y_pred_test.shape == (len(X_test), len(classes))
    # Check probabilities sum to 1 (approx)
    row_sums = y_pred_test.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."
    assert (y_pred_test >= 0).all() and (
        y_pred_test <= 1
    ).all(), "Probabilities out of range."
    print("Prediction shape and probability constraints verified.")

    # Calculate Log Loss on Validation set (using the previously fitted model state)
    # We manually reconstruct the prediction on validation for demonstration
    # Note: In a real scenario, we would use the selector's best score, but here we
    # demonstrate utils.calculate_log_loss.

    # To do this correctly without re-fitting, we'd need the candidate preds from the fit step.
    # Since HDME.fit doesn't return them, we will trust the internal selection logic
    # and just demonstrate the utility function with dummy data or the test preds if we had labels.
    # Here, let's just test the utility function with random valid data.
    dummy_true = np.random.randint(0, len(classes), 10)
    dummy_pred = np.random.rand(10, len(classes))
    loss = utils.calculate_log_loss(dummy_true, dummy_pred)
    print(f"Utils Log Loss Check (Dummy Data): {loss:.4f}")
    assert isinstance(loss, float)

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print("\n--- 6. Generating Submission ---")

    submission_path = "./working/demo_submission.csv"
    utils.format_submission(test_ids, classes, y_pred_test, output_path=submission_path)

    assert os.path.exists(submission_path)
    df_sub = pd.read_csv(submission_path)
    assert df_sub.shape == (len(X_test), len(classes) + 1)  # +1 for id column
    assert "id" in df_sub.columns
    print(f"Submission file generated at {submission_path}")

    # Restore config
    config.LDA_SHRINKAGE_CANDIDATES = original_shrinkage

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
