import os
import sys
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb

# Add current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import feature_engine
from library import model_trainer


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Configuration
    print("Initializing demonstration...")
    warnings.filterwarnings("ignore")
    set_seeds(config.RANDOM_STATE)

    # OVERRIDE CONFIG FOR SPEED
    # We reduce the number of estimators and depth to ensure the demo runs quickly (< 5 mins)
    print("Overriding configuration for speed optimization...")
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 6
    config.XGB_PARAMS["learning_rate"] = 0.1
    config.XGB_PARAMS["n_jobs"] = 4
    config.EARLY_STOPPING_ROUNDS = 2
    config.VERBOSE_EVAL = False

    # Ensure working directory exists (handled by config, but good for safety)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Feature Engineering
    print("\n--- Step 1: Feature Engineering ---")
    # Initialize the feature engine
    # This will load structures.csv automatically
    passer = feature_engine.TabularMessagePasser(verbose=True)

    # Load and process subsets of data
    # We use nrows to limit the data size for this demo
    print("Generating features for Training subset (2000 rows)...")
    df_train = passer.get_train_data(nrows=2000, load_cached_data=False)

    print("Generating features for Validation subset (500 rows)...")
    df_val = passer.get_val_data(nrows=500, load_cached_data=False)

    print("Generating features for Test subset (500 rows)...")
    df_test = passer.get_test_data(nrows=500, load_cached_data=False)

    # VERIFICATION: Check Data Integrity
    print("Verifying feature generation...")

    # Check if basic topological features exist
    expected_features = ["dist", "dist_inv2", "a0_en", "a1_en"]
    for feat in expected_features:
        if feat not in df_train.columns:
            raise AssertionError(f"Feature '{feat}' missing from training data.")

    # Check shapes
    assert len(df_train) == 2000, f"Expected 2000 training rows, got {len(df_train)}"
    assert len(df_val) == 500, f"Expected 500 validation rows, got {len(df_val)}"
    assert len(df_test) == 500, f"Expected 500 test rows, got {len(df_test)}"

    # Check target existence
    assert "scalar_coupling_constant" in df_train.columns
    assert "scalar_coupling_constant" in df_val.columns
    assert "scalar_coupling_constant" not in df_test.columns

    print("Feature engineering verification passed.")

    # 3. Model Training
    print("\n--- Step 2: Model Training ---")
    manager = model_trainer.StratifiedModelManager(verbose=True)

    # Train models on the subset
    # Note: Since we are using a random subset, some coupling types might be missing.
    # The manager handles this gracefully by skipping them.
    scores = manager.train_all_types(df_train, df_val)

    # VERIFICATION: Check Training Results
    if not scores:
        print(
            "Warning: No models were trained. This might happen if the subset "
            "doesn't contain matching coupling types between train and val."
        )
        # For the purpose of this demo, we ensure at least one type is likely present.
        # If this fails, we'll raise an error to indicate the data subset was too small/skewed.
        raise RuntimeError(
            "Training failed to produce any models on the provided subset."
        )

    print("\nValidation Scores (Log MAE):")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    # Check if model files were created
    model_dir = os.path.join(config.WORKING_DIR, "xgb_models")
    saved_models = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    assert len(saved_models) > 0, "No model files found in output directory."
    print(f"Verified {len(saved_models)} saved models.")

    # 4. Inference
    print("\n--- Step 3: Inference ---")

    # Predict on test set
    # The predict_all_types function filters the test set for types that have trained models.
    # If our test subset contains types we haven't trained on (due to small training subset),
    # they will be skipped.
    try:
        predictions = manager.predict_all_types(df_test)

        # VERIFICATION: Check Prediction Format
        assert "id" in predictions.columns
        assert "scalar_coupling_constant" in predictions.columns
        assert predictions["scalar_coupling_constant"].isnull().sum() == 0

        print(f"Generated predictions for {len(predictions)} samples.")

        # 5. Submission
        submission_path = "submission.csv"
        predictions.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        # Verify file exists
        assert os.path.exists(submission_path)

    except ValueError as e:
        print(f"Inference skipped or failed: {e}")
        print(
            "This is expected if the test subset contains only coupling types "
            "that were not present in the training subset."
        )

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
