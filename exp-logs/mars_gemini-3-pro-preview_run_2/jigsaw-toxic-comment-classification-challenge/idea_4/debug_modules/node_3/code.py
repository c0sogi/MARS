import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil
import warnings
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data_processing import load_raw_data, get_tfidf_features, get_mlm_dataset
from library.models import LinearModelWrapper, CustomTransformer
from library.tapt_trainer import run_tapt
from library.supervised_trainer import run_supervised_training
from library.ensemble import (
    optimize_blending_weights,
    blend_predictions,
    create_submission,
    get_val_targets,
    get_test_ids,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Demonstration Script ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Demonstration
    # -------------------------------------------------------------------------
    print("Step 1: Configuring environment for fast execution...")

    # Override Config parameters to ensure the script finishes quickly
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 500  # Increased to 500 to ensure class coverage
    Config.TAPT_PARAMS["epochs"] = 1
    Config.TAPT_PARAMS["save_steps"] = 10
    Config.TRAIN_PARAMS["epochs"] = 1
    Config.TRAIN_PARAMS["batch_size"] = 8
    Config.TRAIN_PARAMS["val_check_interval"] = 1.0  # Validate only at end of epoch

    # Use a tiny model for demonstration to avoid large downloads and long compute
    DEMO_MODEL_NAME = "prajjwal1/bert-tiny"

    # Define temporary working paths for this demo
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config paths to point to demo directory to avoid messing with real artifacts
    Config.WORKING_DIR = DEMO_DIR
    Config.TFIDF_CACHE_DIR = os.path.join(DEMO_DIR, "tfidf_cache")
    Config.VAL_PREDS_PATH = os.path.join(DEMO_DIR, "val_predictions.pkl")
    Config.TEST_PREDS_PATH = os.path.join(DEMO_DIR, "test_predictions.pkl")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    seed_everything(Config.SEED)
    print("Configuration updated successfully.\n")

    # -------------------------------------------------------------------------
    # 2. Data Processing Demonstration
    # -------------------------------------------------------------------------
    print("Step 2: Demonstrating Data Processing...")

    # A. Load Raw Data
    train_df, val_df, test_df = load_raw_data(debug=Config.DEBUG)
    print(
        f"Loaded data shapes (Debug Mode): Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}"
    )

    assert len(train_df) == Config.DEBUG_SAMPLES, "Train DF size mismatch"
    assert len(val_df) == Config.DEBUG_SAMPLES, "Val DF size mismatch"

    # B. TF-IDF Features
    # We force re-computation by setting load_cached_data=False (or ensuring cache dir is empty)
    if os.path.exists(Config.TFIDF_CACHE_DIR):
        shutil.rmtree(Config.TFIDF_CACHE_DIR)

    X_train_tfidf, X_val_tfidf, X_test_tfidf = get_tfidf_features(
        train_df["comment_text"],
        val_df["comment_text"],
        test_df["comment_text"],
        load_cached_data=False,
    )

    print(f"TF-IDF Shapes: Train {X_train_tfidf.shape}, Val {X_val_tfidf.shape}")
    assert X_train_tfidf.shape[0] == Config.DEBUG_SAMPLES
    assert X_val_tfidf.shape[0] == Config.DEBUG_SAMPLES
    print("Data processing checks passed.\n")

    # -------------------------------------------------------------------------
    # 3. Linear Model Demonstration
    # -------------------------------------------------------------------------
    print("Step 3: Demonstrating Linear Model Wrapper...")

    linear_model = LinearModelWrapper(params=Config.LINEAR_PARAMS)

    # Fit model
    y_train = train_df[Config.LABEL_COLS].values
    linear_model.fit(X_train_tfidf, y_train)

    # Predict
    val_preds_linear = linear_model.predict_proba(X_val_tfidf)
    test_preds_linear = linear_model.predict_proba(X_test_tfidf)

    print(f"Linear Model Val Preds Shape: {val_preds_linear.shape}")
    assert val_preds_linear.shape == (Config.DEBUG_SAMPLES, Config.NUM_LABELS)
    assert np.all(
        (val_preds_linear >= 0) & (val_preds_linear <= 1)
    ), "Probabilities out of bounds"

    # Save linear model
    linear_model_path = os.path.join(DEMO_DIR, "linear_model.pkl")
    linear_model.save(linear_model_path)
    assert os.path.exists(linear_model_path)
    print("Linear model trained and saved successfully.\n")

    # -------------------------------------------------------------------------
    # 4. TAPT (Task-Adaptive Pretraining) Demonstration
    # -------------------------------------------------------------------------
    print("Step 4: Demonstrating TAPT (Masked Language Modeling)...")

    tapt_output_path = os.path.join(DEMO_DIR, "tapt_weights")

    # Run TAPT
    # This will train a tiny BERT model on the text for 1 epoch
    run_tapt(
        model_name=DEMO_MODEL_NAME,
        output_path=tapt_output_path,
        debug=True,
        epochs=1,
        batch_size=8,
        mlm_probability=0.15,
    )

    assert os.path.exists(
        os.path.join(tapt_output_path, "config.json")
    ), "TAPT config not saved"
    # Check for either pytorch_model.bin (legacy) or model.safetensors (new default)
    weights_path_bin = os.path.join(tapt_output_path, "pytorch_model.bin")
    weights_path_safe = os.path.join(tapt_output_path, "model.safetensors")

    assert os.path.exists(weights_path_bin) or os.path.exists(
        weights_path_safe
    ), f"TAPT weights not saved. Checked {weights_path_bin} and {weights_path_safe}"
    print("TAPT execution completed successfully.\n")

    # -------------------------------------------------------------------------
    # 5. Supervised Training Demonstration
    # -------------------------------------------------------------------------
    print("Step 5: Demonstrating Supervised Fine-Tuning...")

    supervised_model_path = os.path.join(DEMO_DIR, "supervised_model.bin")
    val_preds_path = os.path.join(DEMO_DIR, "val_preds_transformer.npy")
    test_preds_path = os.path.join(DEMO_DIR, "test_preds_transformer.npy")

    # We use the weights from TAPT (tapt_output_path) as initialization
    # Note: run_supervised_training handles data loading internally based on Config
    val_preds_trans, test_preds_trans = run_supervised_training(
        model_name=DEMO_MODEL_NAME,  # Using tiny model for speed
        pretrained_path=tapt_output_path,
        save_model_path=supervised_model_path,
        val_preds_save_path=val_preds_path,
        test_preds_save_path=test_preds_path,
        debug=True,
    )

    print(f"Transformer Val Preds Shape: {val_preds_trans.shape}")
    assert val_preds_trans.shape == (Config.DEBUG_SAMPLES, Config.NUM_LABELS)
    assert os.path.exists(supervised_model_path), "Best model checkpoint not found"
    print("Supervised training completed successfully.\n")

    # -------------------------------------------------------------------------
    # 6. Ensemble Demonstration
    # -------------------------------------------------------------------------
    print("Step 6: Demonstrating Ensemble Optimization...")

    # Get Ground Truth (subsetted manually because get_val_targets loads full CSV)
    # Since we ran in debug mode, we need the first N targets
    full_val_targets = get_val_targets()
    val_targets = full_val_targets[: Config.DEBUG_SAMPLES]

    preds_list = [val_preds_linear, val_preds_trans]

    # Optimize weights
    weights = optimize_blending_weights(val_targets, preds_list)

    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0)

    # Blend Test Predictions
    test_preds_list = [test_preds_linear, test_preds_trans]
    final_test_preds = blend_predictions(test_preds_list, weights)

    print(f"Final Blended Test Preds Shape: {final_test_preds.shape}")

    # Create Submission
    # Again, get_test_ids loads full CSV, we need subset
    full_test_ids = get_test_ids()
    test_ids = full_test_ids[: Config.DEBUG_SAMPLES]

    create_submission(test_ids, final_test_preds, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH)

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape == (
        Config.DEBUG_SAMPLES,
        Config.NUM_LABELS + 1,
    )  # +1 for id column
    assert list(sub_df.columns) == ["id"] + Config.LABEL_COLS

    print("Ensemble and submission generation completed successfully.\n")

    print("=== All Demonstrations Passed ===")


if __name__ == "__main__":
    main()
