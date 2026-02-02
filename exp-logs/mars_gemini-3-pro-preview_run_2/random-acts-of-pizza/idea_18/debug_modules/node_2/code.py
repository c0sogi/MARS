import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_data
from library.features import UserPersonaTransformer, MetadataTransformer
import library.trainer  # Import module to patch load_data

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_feature_transformers():
    """
    Demonstrates and validates the UserPersonaTransformer and MetadataTransformer
    using synthetic data.
    """
    print("\n=== Demonstrating Feature Transformers ===")

    # ---------------------------------------------------------
    # 1. Test UserPersonaTransformer (View 2)
    # ---------------------------------------------------------
    print("Testing UserPersonaTransformer...")

    # Create dummy data simulating subreddit lists
    # The transformer expects a column with space-separated strings or lists (handled by previous steps)
    # Based on data_loader.py, 'subreddit_text' is space-separated string.
    dummy_data = pd.DataFrame(
        {
            Config.SUBREDDIT_COL: [
                "funny pics gaming",
                "askreddit news",
                "pics aww",
                "science technology",
                "gaming funny",
            ]
        }
    )

    # Instantiate transformer with small n_components
    n_components = 2
    persona_transformer = UserPersonaTransformer(
        subreddit_col=Config.SUBREDDIT_COL,
        n_components=n_components,
        min_df=1,
        random_state=42,
    )

    # Fit and Transform
    X_persona = persona_transformer.fit_transform(dummy_data)

    # Validation
    assert isinstance(X_persona, np.ndarray), "Output should be a numpy array"
    assert X_persona.shape == (
        5,
        n_components,
    ), f"Shape mismatch: expected (5, {n_components}), got {X_persona.shape}"

    # Check L2 Normalization (norms should be approx 1.0)
    norms = np.linalg.norm(X_persona, axis=1)
    assert np.allclose(norms, 1.0), f"Vectors are not L2 normalized. Norms: {norms}"

    print("UserPersonaTransformer passed.")

    # ---------------------------------------------------------
    # 2. Test MetadataTransformer (View 3)
    # ---------------------------------------------------------
    print("Testing MetadataTransformer...")

    # Create dummy numerical data
    # Config.NUMERICAL_COLS contains ~10 columns. Let's use a subset for this test.
    test_cols = ["col1", "col2"]
    dummy_meta = pd.DataFrame(
        {
            "col1": [10.0, 0.0, 50.0, 100.0, np.nan],  # Include NaN to test imputer
            "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    meta_transformer = MetadataTransformer(numerical_cols=test_cols, random_state=42)

    # Fit and Transform
    X_meta = meta_transformer.fit_transform(dummy_meta)

    # Validation
    assert isinstance(X_meta, np.ndarray), "Output should be a numpy array"
    assert X_meta.shape == (
        5,
        2,
    ), f"Shape mismatch: expected (5, 2), got {X_meta.shape}"
    assert not np.isnan(X_meta).any(), "Output contains NaNs after transformation"

    # QuantileTransformer output should be roughly normal (or uniform depending on settings)
    # Here we just check it runs and handles NaNs.

    print("MetadataTransformer passed.")


def fast_load_data_wrapper(load_cached_data=True):
    """
    A wrapper around the actual load_data function to subsample the dataset
    for rapid demonstration purposes.
    """
    print("-> [Fast Wrapper] Loading real data and subsampling...")
    # Call the original load_data
    df_train, df_val, df_test = load_data(load_cached_data=load_cached_data)

    # Subsample to a very small number for speed
    n_samples = 50
    df_train_sub = df_train.head(n_samples).copy()
    df_val_sub = df_val.head(n_samples).copy()
    df_test_sub = df_test.head(n_samples).copy()

    print(
        f"-> [Fast Wrapper] Subsampled Train: {len(df_train_sub)}, Val: {len(df_val_sub)}, Test: {len(df_test_sub)}"
    )
    return df_train_sub, df_val_sub, df_test_sub


def demo_training_pipeline():
    """
    Demonstrates the full training pipeline using the patched data loader.
    """
    print("\n=== Demonstrating Training Pipeline ===")

    # Monkey-patch the load_data function in the trainer module
    # This forces the trainer to use our subsampled data
    original_load_data = library.trainer.load_data
    library.trainer.load_data = fast_load_data_wrapper

    try:
        # Run pipeline
        # Note: We use load_cached_data=False to force processing of our subsample
        # (though the wrapper calls load_data which handles cache logic,
        # we want to ensure we get the data to subsample it).
        auc_score = library.trainer.run_training_pipeline(load_cached_data=False)

        print(f"Pipeline execution complete. OOF AUC: {auc_score:.4f}")

        # Verify Submission
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        assert os.path.exists(submission_path), "Submission file was not created."

        df_sub = pd.read_csv(submission_path)
        print(f"Submission file loaded. Shape: {df_sub.shape}")

        # Check columns
        expected_cols = ["request_id", "requester_received_pizza"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Submission columns mismatch. Got {df_sub.columns}"

        # Check values are probabilities
        probs = df_sub["requester_received_pizza"]
        assert (
            probs.min() >= 0.0 and probs.max() <= 1.0
        ), "Predictions are not valid probabilities."

        print("Pipeline verification passed.")

    finally:
        # Restore original function just in case
        library.trainer.load_data = original_load_data


if __name__ == "__main__":
    # 1. Setup Environment
    set_seed(42)

    # 2. Optimize Config for Speed
    print("Configuring environment for fast demonstration...")
    Config.N_FOLDS = 2
    Config.LR_C_CANDIDATES = [1.0]  # Single hyperparam to skip extensive grid search
    Config.BAGGING_N_ESTIMATORS = 2  # Minimal ensemble
    Config.LR_MAX_ITER = 20  # Minimal iterations

    # Redirect working directories to avoid overwriting production cache
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 3. Run Demonstrations
    demo_feature_transformers()
    demo_training_pipeline()

    print("\nAll demonstrations completed successfully.")
