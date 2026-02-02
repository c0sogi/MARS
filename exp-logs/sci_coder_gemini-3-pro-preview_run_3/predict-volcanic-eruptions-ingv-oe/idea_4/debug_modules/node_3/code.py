import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

# Import provided library modules
import library.config as config
import library.feature_engineering as fe
import library.data_processor as dp
import library.model_trainer as mt

# Silence warnings and LightGBM output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def run_demo():
    print("Initializing Volcano Eruption Prediction Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    SEED = 42
    np.random.seed(SEED)

    # Define paths for mini-datasets in the working directory
    WORKING_DIR = "./working"
    os.makedirs(WORKING_DIR, exist_ok=True)

    MINI_TRAIN_META = os.path.join(WORKING_DIR, "mini_train.csv")
    MINI_VAL_META = os.path.join(WORKING_DIR, "mini_val.csv")
    MINI_TEST_META = os.path.join(WORKING_DIR, "mini_test.csv")

    MINI_TRAIN_FEATS = os.path.join(WORKING_DIR, "mini_train_features.parquet")
    MINI_VAL_FEATS = os.path.join(WORKING_DIR, "mini_val_features.parquet")
    MINI_TEST_FEATS = os.path.join(WORKING_DIR, "mini_test_features.parquet")

    MINI_SUBMISSION = os.path.join(WORKING_DIR, "mini_submission.csv")

    # ==========================================
    # 2. Create Data Subsets (Optimization)
    # ==========================================
    print("Creating data subsets for rapid execution...")

    # Load original metadata
    orig_train = pd.read_csv(config.TRAIN_META_PATH)
    orig_val = pd.read_csv(config.VAL_META_PATH)
    orig_test = pd.read_csv(config.TEST_META_PATH)

    # Sample subsets (20 training, 10 validation, 10 test)
    # This ensures the feature engineering step finishes in seconds instead of hours
    subset_train = orig_train.head(20)
    subset_val = orig_val.head(10)
    subset_test = orig_test.head(10)

    # Save mini metadata
    subset_train.to_csv(MINI_TRAIN_META, index=False)
    subset_val.to_csv(MINI_VAL_META, index=False)
    subset_test.to_csv(MINI_TEST_META, index=False)

    print(
        f"Subsets created: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # ==========================================
    # 3. Patch Library Paths
    # ==========================================
    # We need to redirect the feature engineering module to use our mini files.
    # Since the module imports variables from config, we patch the module's namespace directly.

    print("Patching library configuration to use subsets...")
    fe.TRAIN_META_PATH = MINI_TRAIN_META
    fe.VAL_META_PATH = MINI_VAL_META
    fe.TEST_META_PATH = MINI_TEST_META

    fe.TRAIN_FEATURES_PATH = MINI_TRAIN_FEATS
    fe.VAL_FEATURES_PATH = MINI_VAL_FEATS
    fe.TEST_FEATURES_PATH = MINI_TEST_FEATS

    # ==========================================
    # 4. Feature Engineering & Data Loading
    # ==========================================
    print("Generating features and building dataset...")

    # Force regeneration (load_cached_data=False) to ensure we process the mini files
    dataset = dp.build_dataset(load_cached_data=False)

    X_train, y_train = dataset["train"]
    X_val, y_val = dataset["val"]
    X_test, test_ids = dataset["test"]

    # Validation assertions
    print("Validating dataset shapes...")
    assert len(X_train) == 20, f"Expected 20 training samples, got {len(X_train)}"
    assert len(y_train) == 20, "Target size mismatch for training"
    assert len(X_val) == 10, f"Expected 10 validation samples, got {len(X_val)}"
    assert len(X_test) == 10, f"Expected 10 test samples, got {len(X_test)}"
    assert not X_train.isnull().values.any(), "Training features contain NaNs"

    print("Feature generation successful.")

    # ==========================================
    # 5. Model Training
    # ==========================================
    print("Initializing Model Trainer...")

    # Override parameters for speed
    fast_params = config.LGBM_PARAMS.copy()
    fast_params.update(
        {
            "n_estimators": 10,  # Very few trees for demo
            "learning_rate": 0.1,
            "verbose": -1,
            "seed": SEED,
        }
    )

    predictor = mt.EruptionPredictor(params=fast_params)

    print("Training model...")
    predictor.fit(X_train, y_train, X_val, y_val)

    # Verify model was created
    assert predictor.model is not None, "Model failed to initialize after fit()"

    # Check prediction logic on validation set
    val_preds = predictor.predict(X_val)
    assert len(val_preds) == len(X_val), "Prediction shape mismatch"
    assert np.all(np.isfinite(val_preds)), "Predictions contain non-finite values"

    print("Training and validation check successful.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("Generating submission...")

    mt.generate_submission(
        predictor=predictor,
        X_test=X_test,
        test_ids=test_ids,
        output_path=MINI_SUBMISSION,
    )

    # Verify submission file
    assert os.path.exists(MINI_SUBMISSION), "Submission file was not created"

    submission_df = pd.read_csv(MINI_SUBMISSION)
    assert list(submission_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns mismatch"
    assert (
        len(submission_df) == 10
    ), f"Expected 10 submission rows, got {len(submission_df)}"

    print(f"Submission generated successfully at {MINI_SUBMISSION}")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
