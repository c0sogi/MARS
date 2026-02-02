import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch

# Import library modules
from library.config import Config
from library import utils
from library.feature_engineering import FeatureEngineer
from library import dataset
from library import models
from library.engine import CrossValidationRunner


def run_demonstration():
    print("=== Starting Tri-View Stacking Ensemble Demonstration ===\n")

    # -----------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -----------------------------------------------------------------------
    # We modify the configuration dictionaries in-place to ensure models
    # run quickly during this demo.
    print("[1] Configuring hyperparameters for fast execution...")

    # Reduce TF-IDF dimensionality
    Config.TFIDF_MAX_FEATURES = 100

    # Reduce Random Forest complexity
    Config.RF_PARAMS["n_estimators"] = 10

    # Reduce XGBoost complexity
    Config.XGB_PARAMS["n_estimators"] = 10

    # Reduce Transformer training load
    Config.BERT_TRAIN_PARAMS["epochs"] = 1
    Config.BERT_TRAIN_PARAMS["batch_size"] = 4

    # Note: We cannot easily change file paths that are bound as default arguments
    # in the library functions, so we will use the default locations
    # (./working/idea_4 and ./submission).

    # -----------------------------------------------------------------------
    # 2. Testing Utilities and Data Loading
    # -----------------------------------------------------------------------
    print("[2] Testing Data Loading and Utils...")

    # Set seed for reproducibility
    utils.set_seed(42)

    # Load a small subset of training data
    df_train = utils.load_data(Config.TRAIN_PATH, debug=True, n_rows=50)

    # Verify
    assert isinstance(df_train, pd.DataFrame), "load_data should return a DataFrame"
    assert len(df_train) == 50, "Debug mode should limit rows to 50"
    assert Config.TARGET_COL in df_train.columns, "Target column missing in loaded data"
    print("    -> Data loading verified.")

    # -----------------------------------------------------------------------
    # 3. Testing Feature Engineering
    # -----------------------------------------------------------------------
    print("[3] Testing Feature Engineering...")

    fe = FeatureEngineer()

    # Process data in debug mode. We set load_cached_data=False to force computation
    # and verify the logic, ensuring we don't just read old files.
    data_bundle = fe.process_data(load_cached_data=False, debug=True)

    # Verify structure
    for split in ["train", "val", "test"]:
        assert split in data_bundle, f"Missing split '{split}' in processed data"
        assert "lexical" in data_bundle[split]
        assert "style" in data_bundle[split]
        assert "meta" in data_bundle[split]

    # Verify dimensions
    n_train = len(data_bundle["train"]["y"])
    assert data_bundle["train"]["lexical"].shape == (
        n_train,
        Config.TFIDF_MAX_FEATURES,
    ), "Lexical feature shape mismatch"
    assert (
        data_bundle["train"]["style"].shape[0] == n_train
    ), "Style feature row mismatch"
    assert data_bundle["train"]["meta"].shape[0] == n_train, "Meta feature row mismatch"

    print("    -> Feature Engineering verified (Lexical, Style, Meta generated).")

    # -----------------------------------------------------------------------
    # 4. Testing Dataset and DataLoaders (Semantic View)
    # -----------------------------------------------------------------------
    print("[4] Testing PyTorch Dataset and DataLoaders...")

    # Create dataloaders (internally tokenizes text)
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch to verify
    batch = next(iter(train_loader))

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert (
        batch["input_ids"].shape[0] == Config.BERT_TRAIN_PARAMS["batch_size"]
    ), "Batch size mismatch"
    assert batch["input_ids"].shape[1] == Config.MAX_SEQ_LEN, "Sequence length mismatch"

    print("    -> DataLoaders verified.")

    # -----------------------------------------------------------------------
    # 5. Testing Model Instantiation and Training
    # -----------------------------------------------------------------------
    print("[5] Testing Model Training (Level 1 Models)...")

    # A. Lexical Model (Random Forest)
    print("    -> Testing Lexical Model (RF)...")
    rf_model = models.get_lexical_model()
    # Create dummy data matching the config dimensions
    X_dummy = np.random.rand(n_train, Config.TFIDF_MAX_FEATURES)
    y_dummy = data_bundle["train"]["y"]
    rf_model.fit(X_dummy, y_dummy)
    preds_rf = rf_model.predict_proba(X_dummy)
    assert preds_rf.shape == (n_train, 2), "RF prediction shape mismatch"

    # B. Style Model (XGBoost)
    print("    -> Testing Style Model (XGBoost)...")
    xgb_model = models.get_style_model()
    # Style features usually have fixed width based on engineering (approx 9 features + meta)
    # We use the actual generated style features here to be safe
    X_style_dummy = data_bundle["train"]["style"]
    xgb_model.fit(X_style_dummy, y_dummy)
    preds_xgb = xgb_model.predict_proba(X_style_dummy)
    assert preds_xgb.shape == (n_train, 2), "XGB prediction shape mismatch"

    # C. Semantic Model (DistilBERT Fine-Tuner)
    print("    -> Testing Semantic Model (DistilBERT)...")
    # We use the dataloaders created in Step 4
    bert_tuner = models.SemanticFineTuner()
    bert_tuner.fit(train_loader, val_loader)
    probs_bert = bert_tuner.predict_proba(val_loader)
    assert len(probs_bert) == len(val_loader.dataset), "BERT prediction count mismatch"

    print("    -> All Level 1 models verified.")

    # -----------------------------------------------------------------------
    # 6. Testing Full Execution Engine
    # -----------------------------------------------------------------------
    print("[6] Testing Full CrossValidationRunner...")

    # Instantiate runner with explicit debug flags and reduced folds
    runner = CrossValidationRunner(n_folds=2, debug=True)

    # Run the full pipeline (Feature Eng -> CV -> Stacking -> Prediction)
    runner.run()

    # Verify submission file creation
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"    -> Pipeline completed. Submission generated with {len(sub_df)} rows.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    run_demonstration()
