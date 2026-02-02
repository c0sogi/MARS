import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, calculate_log_loss, save_submission
from library.data_loader import load_data
from library.models_statistical import StatisticalExpert
from library.models_neural import TransformerExpert
from library.stacking import MetaLearner
from library.engine import run_full_pipeline


def run_demonstration():
    print("=== Starting Code Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and resource efficiency
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 60  # Small subset for quick training
    Config.EPOCHS = 1  # Single epoch to verify training loop
    Config.N_FOLDS = 2  # Minimum folds for CV
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8

    # Use a tiny model to avoid large downloads and slow compute during demo
    # This validates the pipeline logic without the overhead of Large models
    TINY_MODEL = "prajjwal1/bert-tiny"
    Config.MODEL_DEBERTA = TINY_MODEL
    Config.MODEL_ROBERTA = TINY_MODEL

    # Ensure working directory is clean for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print("Configuration updated: Debug Mode=True, Model=bert-tiny, Epochs=1")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test calculate_log_loss
    y_true = [0, 1, 2]  # EAP, HPL, MWS
    # Preds: Perfect prediction for 0, Split for 1, Wrong for 2
    y_pred = np.array([[0.99, 0.005, 0.005], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])

    loss = calculate_log_loss(y_true, y_pred)
    print(f"Calculated Log Loss: {loss:.4f}")

    # Assertions
    assert isinstance(loss, float), "Log loss should return a float"
    assert loss > 0, "Log loss should be positive"

    # Test save_submission
    dummy_ids = ["id001", "id002", "id003"]
    dummy_probs = np.array([[0.3, 0.3, 0.4], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    dummy_path = os.path.join(Config.WORKING_DIR, "dummy_submission.csv")

    save_submission(dummy_ids, dummy_probs, output_path=dummy_path)
    assert os.path.exists(dummy_path), "Submission file was not created"

    df_sub = pd.read_csv(dummy_path)
    assert list(df_sub.columns) == [
        "id",
        "EAP",
        "HPL",
        "MWS",
    ], "Incorrect submission columns"
    assert len(df_sub) == 3, "Incorrect number of rows in submission"
    print("Utility functions verified.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loader
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loader...")

    # Load data in debug mode
    train_df, val_df, test_df = load_data(load_cached_data=False, debug=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # Assertions
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train set size mismatch"
    assert "log_char_len" in train_df.columns, "Meta-feature 'log_char_len' missing"
    assert "target" in train_df.columns, "Target column missing in train"
    assert train_df["target"].dtype in [
        np.int64,
        np.int32,
        int,
    ], "Target should be integer encoded"
    print("Data Loader verified.")

    # -------------------------------------------------------------------------
    # 4. Verify Statistical Expert
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Statistical Expert...")

    stat_model = StatisticalExpert()

    # Fit
    stat_model.fit(train_df["text"], train_df["target"])

    # Predict
    stat_preds = stat_model.predict_proba(val_df["text"])

    # Assertions
    assert stat_preds.shape == (
        len(val_df),
        3,
    ), "Statistical expert output shape mismatch"
    assert np.allclose(stat_preds.sum(axis=1), 1.0), "Probabilities must sum to 1"
    print("Statistical Expert verified.")

    # -------------------------------------------------------------------------
    # 5. Verify Neural Expert
    # -------------------------------------------------------------------------
    print(f"\n[5] Verifying Neural Expert ({Config.MODEL_DEBERTA})...")

    neural_model = TransformerExpert(model_name=Config.MODEL_DEBERTA)

    # Fit (using small debug data)
    # Using train for both train and val arguments just for API verification
    neural_model.fit(
        train_texts=train_df["text"].values,
        train_labels=train_df["target"].values,
        val_texts=val_df["text"].values,
        val_labels=val_df["target"].values,
    )

    # Predict
    neural_preds = neural_model.predict_proba(val_df["text"].values)

    # Assertions
    assert neural_preds.shape == (len(val_df), 3), "Neural expert output shape mismatch"
    # Softmax output should sum to 1
    assert np.allclose(
        neural_preds.sum(axis=1), 1.0, atol=1e-5
    ), "Neural probs must sum to 1"
    print("Neural Expert verified.")

    # -------------------------------------------------------------------------
    # 6. Verify Meta-Learner (Stacking)
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Meta-Learner...")

    meta_learner = MetaLearner()

    # Prepare fake level 1 features
    # We need predictions from base models + meta features
    # Let's use the predictions we just generated

    # Ensure inputs are consistent length (using val_df size)
    base_preds = [stat_preds, neural_preds, neural_preds]  # Simulating 3 models
    meta_feats = val_df["log_char_len"].values
    targets = val_df["target"].values

    X_meta = meta_learner.prepare_level1_features(base_preds, meta_feats)

    # Expected shape: 3 models * 3 classes + 1 meta feature = 10 columns
    expected_cols = 3 * 3 + 1
    assert X_meta.shape == (
        len(val_df),
        expected_cols,
    ), f"Meta feature shape mismatch. Got {X_meta.shape}"

    # Fit
    meta_learner.fit(X_meta, targets)

    # Predict
    meta_probs = meta_learner.predict_proba(X_meta)

    assert meta_probs.shape == (len(val_df), 3), "Meta learner output shape mismatch"
    print("Meta-Learner verified.")

    # -------------------------------------------------------------------------
    # 7. Verify Full Pipeline Engine
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Full Pipeline Engine...")

    # This runs the end-to-end process: Load -> CV -> Train Base -> Stack -> Submit
    # We use n_folds=2 and debug=True for speed

    try:
        run_full_pipeline(debug=True, n_folds=2)
        print("Pipeline execution successful.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise e

    # Verify final submission exists
    if os.path.exists(Config.SUBMISSION_FILE):
        print(f"Submission file found at {Config.SUBMISSION_FILE}")
        final_df = pd.read_csv(Config.SUBMISSION_FILE)
        print(f"Submission shape: {final_df.shape}")
        assert final_df.shape[1] == 4, "Submission should have 4 columns"
    else:
        raise FileNotFoundError("Submission file was not generated by the pipeline.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
