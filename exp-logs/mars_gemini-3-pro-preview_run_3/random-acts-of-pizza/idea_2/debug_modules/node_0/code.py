import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_data_splits
from library.features import FeatureProcessor, process_and_cache_data
from library.models import MultiViewEnsemble


def main():
    # 1. Setup and Configuration for Speed
    # We modify the Config class attributes directly to ensure the demo runs quickly.
    print(">>> Setting up configuration for rapid demonstration...")
    set_seed(42)
    logger = setup_logger("demo_script")

    # Reduce Random Forest complexity
    Config.RF_PARAMS["n_estimators"] = 10
    Config.RF_PARAMS["max_depth"] = 4

    # Reduce Logistic Regression complexity
    Config.LR_PARAMS["max_iter"] = 50

    # Reduce Feature dimensions for speed
    Config.TFIDF_MAX_FEATURES = 500

    # Use a temporary working directory for this demo to avoid conflicts
    # (Though in a real run we would use the default)
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORKING_DIR
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("\n>>> Loading data splits (subsampled)...")
    # Load only 50 samples per split to ensure speed
    train_df, val_df, test_df = load_data_splits(max_samples=50)

    # Verification
    assert len(train_df) == 50, "Train set should have 50 samples"
    assert len(val_df) == 50, "Val set should have 50 samples"
    assert len(test_df) == 50, "Test set should have 50 samples"
    assert Config.TARGET_COL in train_df.columns, "Target column missing in train"
    print("Data loading verified.")

    # 3. Feature Processing (Manual Demonstration)
    print("\n>>> Demonstrating FeatureProcessor usage...")
    processor = FeatureProcessor()

    # Fit on training data
    print("Fitting processor...")
    processor.fit(train_df)

    # Transform training data
    print("Transforming training data...")
    X_train_dict = processor.transform(train_df)

    # Verify feature dictionary structure
    assert isinstance(X_train_dict, dict), "Transform should return a dictionary"
    assert "tfidf" in X_train_dict, "Missing TF-IDF features"
    assert "embedding" in X_train_dict, "Missing Embedding features"
    assert "dense" in X_train_dict, "Missing Dense features"

    # Verify shapes
    n_samples = len(train_df)
    assert X_train_dict["tfidf"].shape[0] == n_samples
    assert X_train_dict["embedding"].shape[0] == n_samples
    assert X_train_dict["dense"].shape[0] == n_samples

    print(f"TF-IDF shape: {X_train_dict['tfidf'].shape}")
    print(f"Embedding shape: {X_train_dict['embedding'].shape}")
    print(f"Dense shape: {X_train_dict['dense'].shape}")
    print("Feature processing verified.")

    # 4. Pipeline Integration (Process and Cache)
    print("\n>>> Demonstrating process_and_cache_data pipeline...")
    # This function handles fitting, transforming, and caching for all splits
    # We force load_cached_data=False to ensure the code actually runs
    (X_train_p, y_train_p), (X_val_p, y_val_p), (X_test_p, y_test_p) = (
        process_and_cache_data(train_df, val_df, test_df, load_cached_data=False)
    )

    # Verify that the pipeline outputs match the manual process logic
    assert y_train_p is not None
    assert y_val_p is not None
    assert y_test_p is None
    assert X_train_p["dense"].shape == X_train_dict["dense"].shape
    print("Pipeline integration verified.")

    # 5. Model Training and Prediction
    print("\n>>> Demonstrating MultiViewEnsemble model...")
    model = MultiViewEnsemble()

    # Fit the model
    # Note: process_and_cache_data returns y as numpy arrays, which is what we need
    print("Training ensemble model...")
    model.fit(X_train_p, y_train_p, X_val_dict=X_val_p, y_val=y_val_p)

    # Predict Probabilities
    print("Generating predictions...")
    train_probs = model.predict_proba(X_train_p)
    val_probs = model.predict_proba(X_val_p)
    test_probs = model.predict_proba(X_test_p)

    # Verify Predictions
    assert len(train_probs) == 50
    assert len(test_probs) == 50
    assert (train_probs >= 0).all() and (
        train_probs <= 1
    ).all(), "Probabilities must be [0, 1]"

    # Calculate AUC manually to verify
    try:
        train_auc = roc_auc_score(y_train_p, train_probs)
        val_auc = roc_auc_score(y_val_p, val_probs)
        print(f"Manual Check - Train AUC: {train_auc:.4f}")
        print(f"Manual Check - Val AUC: {val_auc:.4f}")
    except ValueError:
        # This might happen if the subsample has only one class
        print("Skipping AUC check due to single-class subsample.")

    # Predict Labels (Thresholding)
    test_preds = model.predict(X_test_p, threshold=0.5)
    assert set(np.unique(test_preds)).issubset({0, 1}), "Predictions must be binary"

    print("Model training and prediction verified.")

    # 6. Cleanup (Optional)
    # Removing the demo working directory to keep things clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
        print(f"\nCleaned up temporary directory: {Config.WORKING_DIR}")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
