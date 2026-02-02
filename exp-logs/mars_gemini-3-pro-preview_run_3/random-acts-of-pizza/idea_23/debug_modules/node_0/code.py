import os
import sys
import numpy as np
import pandas as pd
import shutil
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
import library.config
import library.utils
import library.data_processing
import library.models
import library.pipeline


def main():
    print("=== Starting Demonstration of Pizza Request Prediction Pipeline ===\n")

    # -------------------------------------------------------------------------
    # 1. OPTIMIZATION FOR SPEED
    # -------------------------------------------------------------------------
    print("Step 1: Optimizing hyperparameters for fast demonstration...")

    # Patch mutable configuration dictionaries (updates reflect across modules)
    library.config.RF_PARAMS["n_estimators"] = 5
    library.config.RF_PARAMS["n_jobs"] = 1  # Reduce overhead for small demo

    library.config.XGB_PARAMS["n_estimators"] = 5
    library.config.XGB_PARAMS["n_jobs"] = 1

    library.config.LR_PARAMS["max_iter"] = 10

    library.config.TFIDF_PARAMS["max_features"] = 50  # Drastically reduce vocab size

    # Patch immutable constants in the modules where they are used
    # Note: Since 'from X import Y' was used in library files, we must patch the local namespace of those modules
    library.data_processing.PCA_COMPONENTS = 5
    library.pipeline.N_FOLDS = 2

    # Redirect cache to a demo directory to avoid messing with real working files
    DEMO_CACHE_DIR = "./working/demo_cache"
    library.config.CACHE_DIR = DEMO_CACHE_DIR
    library.utils.CACHE_DIR = DEMO_CACHE_DIR

    # Ensure clean slate
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    print("Hyperparameters optimized.\n")

    # -------------------------------------------------------------------------
    # 2. VERIFY UTILITIES
    # -------------------------------------------------------------------------
    print("Step 2: Verifying Utility Functions...")

    # Test Seed
    library.utils.set_seed(42)
    rand_val1 = np.random.rand()
    library.utils.set_seed(42)
    rand_val2 = np.random.rand()
    assert rand_val1 == rand_val2, "set_seed failed to produce reproducible results."

    # Test Caching
    test_data = np.array([1, 2, 3, 4, 5])
    library.utils.save_to_cache(test_data, "test_array")
    loaded_data = library.utils.load_from_cache("test_array")

    assert loaded_data is not None, "Failed to load cached data."
    assert np.array_equal(
        test_data, loaded_data
    ), "Loaded data does not match saved data."
    print("Utils verified: Seeding and Caching work correctly.\n")

    # -------------------------------------------------------------------------
    # 3. VERIFY DATA PROCESSING
    # -------------------------------------------------------------------------
    print("Step 3: Verifying Data Processing Pipeline...")

    processor = library.data_processing.DataProcessor()

    # Force processing (ignore cache) to test logic
    data_dict = processor.process_data(load_cached_data=False)

    # Validation
    expected_keys = [
        "y_train",
        "y_val",
        "test_ids",
        "X_train_lexical",
        "X_val_lexical",
        "X_test_lexical",
        "X_train_behavioral",
        "X_val_behavioral",
        "X_test_behavioral",
        "X_train_semantic",
        "X_val_semantic",
        "X_test_semantic",
        "X_train_manifold",
        "X_val_manifold",
        "X_test_manifold",
        "X_train_contextual",
        "X_val_contextual",
        "X_test_contextual",
    ]

    for key in expected_keys:
        assert key in data_dict, f"Missing key in processed data: {key}"

    # Check dimensions
    n_train = len(data_dict["y_train"])
    n_val = len(data_dict["y_val"])
    n_test = len(data_dict["test_ids"])

    print(f"Data Shapes -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

    assert data_dict["X_train_lexical"].shape[0] == n_train
    assert data_dict["X_val_semantic"].shape[0] == n_val
    assert data_dict["X_test_manifold"].shape[0] == n_test

    print("Data Processing verified.\n")

    # -------------------------------------------------------------------------
    # 4. VERIFY MODEL REGISTRY
    # -------------------------------------------------------------------------
    print("Step 4: Verifying Model Registry...")

    base_models = library.models.ModelRegistry.create_base_models()
    expected_models = [
        "lexical_bagger",
        "community_bagger",
        "semantic_booster",
        "semantic_bagger",
        "manifold_neighbor",
        "metadata_anchor",
    ]

    assert isinstance(base_models, dict)
    for m in expected_models:
        assert m in base_models, f"Model Registry missing: {m}"

    meta_learner = library.models.ModelRegistry.get_meta_learner()
    assert meta_learner is not None

    print("Model Registry verified.\n")

    # -------------------------------------------------------------------------
    # 5. VERIFY PIPELINE (TRAINING & SUBMISSION)
    # -------------------------------------------------------------------------
    print("Step 5: Executing Stacking Trainer Pipeline...")

    trainer = library.pipeline.StackingTrainer()

    # Inject the already processed data to save time (though load_data calls process_data internally)
    trainer.data = data_dict

    # A. Cross-Validation
    print("Running Cross-Validation (2 Folds)...")
    trainer.run_cv()

    # Check if meta-learner was trained (simple check on attributes)
    # LogisticRegression should have 'coef_' after fitting
    assert hasattr(
        trainer.meta_learner, "coef_"
    ), "Meta-learner was not fitted during CV."

    # B. Retraining
    print("Retraining Base Models on Full Data...")
    trainer.retrain_final()

    # Check if a base model was fitted (e.g., Random Forest has 'estimators_')
    rf_model = trainer.base_models["lexical_bagger"]
    assert hasattr(
        rf_model, "estimators_"
    ), "Base model (RF) was not fitted during retraining."

    # C. Submission Generation
    print("Generating Submission...")
    # Override submission path for demo
    DEMO_SUBMISSION_PATH = "./working/demo_submission/submission.csv"
    library.pipeline.SUBMISSION_PATH = DEMO_SUBMISSION_PATH

    trainer.generate_submission()

    # -------------------------------------------------------------------------
    # 6. FINAL OUTPUT VALIDATION
    # -------------------------------------------------------------------------
    print("Step 6: Validating Submission File...")

    assert os.path.exists(DEMO_SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(DEMO_SUBMISSION_PATH)

    # Check Columns
    assert "request_id" in df_sub.columns
    assert "requester_received_pizza" in df_sub.columns

    # Check Rows (Test set size is 1162)
    assert len(df_sub) == 1162, f"Expected 1162 rows, got {len(df_sub)}"

    # Check Values (Probabilities between 0 and 1)
    probs = df_sub["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Predictions are not valid probabilities."

    print("\n=== Demonstration Completed Successfully ===")
    print(f"Submission generated at: {DEMO_SUBMISSION_PATH}")
    print(df_sub.head())


if __name__ == "__main__":
    main()
