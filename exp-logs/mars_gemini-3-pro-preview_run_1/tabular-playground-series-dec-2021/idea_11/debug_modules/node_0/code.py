import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_processor import process_data
from library.feature_learner import SupervisedProjector
from library.model_handler import ModelTrainer


def main():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Setup & Reproducibility
    set_seed(42)
    working_dir = "./working"
    demo_train_path = os.path.join(working_dir, "demo_train.csv")
    demo_test_path = os.path.join(working_dir, "demo_test.csv")
    demo_cache_dir = os.path.join(working_dir, "demo_cache")
    demo_submission_dir = os.path.join(working_dir, "demo_submission")

    # Ensure working directories exist
    os.makedirs(working_dir, exist_ok=True)

    # 2. Create Subsampled Data for Speed
    # We read only the first 2000 rows to make the demo run in seconds
    print("Creating subsampled datasets...")
    try:
        df_train_full = pd.read_csv("./metadata/train.csv", nrows=2000)
        df_test_full = pd.read_csv("./metadata/test.csv", nrows=500)

        df_train_full.to_csv(demo_train_path, index=False)
        df_test_full.to_csv(demo_test_path, index=False)
        print(f"  Saved demo train: {df_train_full.shape}")
        print(f"  Saved demo test: {df_test_full.shape}")
    except FileNotFoundError as e:
        print(f"Error: Metadata files not found. {e}")
        sys.exit(1)

    # 3. Monkey Patch Config
    # We override the Config paths to point to our demo data
    print("Configuring paths...")
    Config.TRAIN_PATH = demo_train_path
    Config.TEST_PATH = demo_test_path
    Config.CACHE_DIR = demo_cache_dir
    Config.SUBMISSION_DIR = demo_submission_dir

    # Clean cache if exists to ensure we process data
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)

    # 4. Demonstrate Data Processor
    print("\n[1] Testing Data Processor...")
    # process_data handles loading, cleaning, and feature engineering
    train_df, test_df = process_data(load_cached_data=False)

    # Validation
    assert train_df.shape[0] == 2000, "Train DataFrame row count mismatch"
    assert test_df.shape[0] == 500, "Test DataFrame row count mismatch"

    # Check for engineered features
    expected_features = ["Hydrology_Distance", "Soil_Type_Index", "Aspect_Sin"]
    for feat in expected_features:
        if feat not in train_df.columns:
            raise AssertionError(f"Feature engineering failed: {feat} missing.")

    print("  Data processing successful. Engineered features verified.")

    # 5. Demonstrate Feature Learner (LDA)
    print("\n[2] Testing Feature Learner (SupervisedProjector)...")

    # Prepare data for LDA
    target_col = "Cover_Type"
    drop_cols = [target_col, "Id"]

    # Filter columns that exist
    X_cols = [c for c in train_df.columns if c not in drop_cols]
    X = train_df[X_cols]
    y = train_df[target_col]

    # Initialize Projector
    # We use n_components=2 for this demo
    projector = SupervisedProjector(n_components=2)

    # Fit and Transform
    X_proj = projector.fit_transform(X, y)

    # Validation
    assert X_proj.shape[0] == 2000, "Projected data row count mismatch"
    assert X_proj.shape[1] <= 2, "Projected data column count mismatch"

    # Transform test set
    X_test_proj = projector.transform(test_df[X_cols])
    assert X_test_proj.shape[0] == 500, "Test projection row count mismatch"

    print(f"  LDA Projection successful. Output shape: {X_proj.shape}")

    # 6. Demonstrate Model Handler (XGBoost)
    print("\n[3] Testing Model Handler (ModelTrainer)...")

    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Initialize Trainer with fast parameters
    # Overriding n_estimators to 10 for speed
    fast_params = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "max_depth": 3,
        "device": "cuda",  # Using GPU as available
        "tree_method": "hist",
    }
    trainer = ModelTrainer(params=fast_params)

    # Train
    print("  Training XGBoost model...")
    trainer.train(X_train, y_train, X_val, y_val)

    # Predict
    preds = trainer.predict(X_val)
    probs = trainer.predict_proba(X_val)

    # Validation
    assert len(preds) == len(y_val), "Prediction length mismatch"
    assert probs.shape == (len(y_val), len(le.classes_)), "Probability shape mismatch"
    # Check if probabilities sum to ~1
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1"

    acc = accuracy_score(y_val, preds)
    print(f"  Model training successful. Validation Accuracy: {acc:.4f}")

    # 7. Demonstrate Utils (Submission)
    print("\n[4] Testing Utils (Save Submission)...")

    # Generate predictions for the test set
    test_preds_enc = trainer.predict(test_df[X_cols])
    test_preds = le.inverse_transform(test_preds_enc)

    test_ids = test_df["Id"]

    # Save submission
    save_submission(test_ids, test_preds)

    # Validation
    submission_path = os.path.join(Config.SUBMISSION_DIR, Config.SUBMISSION_FILE)
    if not os.path.exists(submission_path):
        raise AssertionError("Submission file was not created.")

    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (500, 2), "Submission file shape mismatch"
    assert list(sub_df.columns) == ["Id", "Cover_Type"], "Submission columns mismatch"

    print("  Submission file generated and verified.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
