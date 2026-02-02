import os
import shutil
import warnings
import pandas as pd
import numpy as np

# Import library components
from library.inference import Predictor
from library.features import FeaturePipeline
from library.training import StackingTrainer
from library.models import ModelFactory
from library.utils import set_seed
from library.config import CACHE_DIR, SUBMISSION_PATH

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def clean_environment():
    """Cleans up cache and submission directories to ensure a fresh run."""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(SUBMISSION_PATH):
        os.remove(SUBMISSION_PATH)


def demo_high_level_api():
    """
    Demonstrates the end-to-end usage using the Predictor class.
    This is the standard way to run the solution.
    """
    print("\n" + "=" * 40)
    print("DEMO 1: High-Level API (Predictor)")
    print("=" * 40)

    # Use a small sample size for rapid execution
    sample_size = 50
    print(f"Initializing Predictor with debug_sample_size={sample_size}...")
    predictor = Predictor(debug_sample_size=sample_size)

    print("Running generate_predictions()...")
    predictor.generate_predictions()

    # Validation of the output
    print("\nVerifying submission output...")
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_PATH}")

    df_submission = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_submission.shape}")
    print(df_submission.head())

    # Assertions
    assert df_submission.shape == (
        sample_size,
        2,
    ), f"Expected shape ({sample_size}, 2), got {df_submission.shape}"

    expected_cols = ["request_id", "requester_received_pizza"]
    assert (
        list(df_submission.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_submission.columns)}"

    # Check probability range
    probs = df_submission["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Predictions contain values outside [0, 1] range."

    print("High-Level API Demo Passed Successfully.")


def demo_low_level_api():
    """
    Demonstrates the component-level usage.
    Useful for understanding the data flow and internal logic.
    """
    print("\n" + "=" * 40)
    print("DEMO 2: Low-Level API (Components)")
    print("=" * 40)

    sample_size = 30

    # ---------------------------------------------------------
    # 1. Feature Engineering
    # ---------------------------------------------------------
    print("Step 1: Feature Pipeline Initialization...")
    # Initialize pipeline with small sample
    pipeline = FeaturePipeline(debug_sample_size=sample_size)

    # Load all feature views
    # We set load_cached_data=False to force re-computation for this specific sample size
    # and to demonstrate the feature generation logic.
    print("Generating feature views...")
    meta_tr, meta_val, meta_te = pipeline.get_metadata_view(load_cached_data=False)
    lex_sp_tr, lex_sp_val, lex_sp_te = pipeline.get_lexical_sparse_view(
        load_cached_data=False
    )
    lex_dn_tr, lex_dn_val, lex_dn_te = pipeline.get_lexical_dense_view(
        load_cached_data=False
    )
    beh_sp_tr, beh_sp_val, beh_sp_te = pipeline.get_behavioral_sparse_view(
        load_cached_data=False
    )
    beh_dn_tr, beh_dn_val, beh_dn_te = pipeline.get_behavioral_dense_view(
        load_cached_data=False
    )

    # Verify shapes
    print(f"Metadata Train Shape: {meta_tr.shape}")
    assert meta_tr.shape[0] == sample_size, "Metadata train rows mismatch."
    assert lex_sp_tr.shape[0] == sample_size, "Lexical sparse train rows mismatch."

    # Get Targets
    y_train, y_val = pipeline.get_targets()
    assert len(y_train) == sample_size, "Target train length mismatch."

    # ---------------------------------------------------------
    # 2. Model Configuration
    # ---------------------------------------------------------
    print("\nStep 2: Model Factory Verification...")
    # Verify we can instantiate a model
    rf_model = ModelFactory.get_lexical_sparse_rf(n_estimators=10)
    assert rf_model.n_estimators == 10, "ModelFactory parameter override failed."
    print("ModelFactory instantiated a Random Forest correctly.")

    # ---------------------------------------------------------
    # 3. Stacking Trainer
    # ---------------------------------------------------------
    print("\nStep 3: Stacking Trainer Execution...")

    # Prepare dictionaries as expected by the Trainer
    X_train_dict = {
        "metadata": meta_tr,
        "lexical_sparse": lex_sp_tr,
        "lexical_dense": lex_dn_tr,
        "behavioral_sparse": beh_sp_tr,
        "behavioral_dense": beh_dn_tr,
    }

    X_val_dict = {
        "metadata": meta_val,
        "lexical_sparse": lex_sp_val,
        "lexical_dense": lex_dn_val,
        "behavioral_sparse": beh_sp_val,
        "behavioral_dense": beh_dn_val,
    }

    X_test_dict = {
        "metadata": meta_te,
        "lexical_sparse": lex_sp_te,
        "lexical_dense": lex_dn_te,
        "behavioral_sparse": beh_sp_te,
        "behavioral_dense": beh_dn_te,
    }

    trainer = StackingTrainer(X_train_dict, y_train, X_val_dict, y_val)

    # Run Level 1 CV and Meta Training
    print("Running Level 1 Cross-Validation...")
    trainer.run_cv_and_meta_training()

    assert trainer.meta_learner is not None, "Meta-learner was not trained."

    # Run Final Retraining
    print("Retraining Final Base Models...")
    trainer.train_final_base_models()

    assert len(trainer.final_base_models) == len(
        trainer.model_keys
    ), "Not all base models were retrained."

    # Run Prediction
    print("Generating predictions...")
    preds = trainer.predict(X_test_dict)

    assert len(preds) == sample_size, "Prediction length mismatch."
    print(f"Predictions generated: {preds[:5]}...")

    print("Low-Level API Demo Passed Successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Clean environment before starting
    clean_environment()

    try:
        # Run Demo 1
        demo_high_level_api()

        # Clean environment between demos to avoid cache conflicts with different sample sizes
        clean_environment()

        # Run Demo 2
        demo_low_level_api()

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        raise e
