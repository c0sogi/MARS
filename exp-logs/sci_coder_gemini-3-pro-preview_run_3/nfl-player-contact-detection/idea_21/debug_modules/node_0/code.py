import os
import pandas as pd
import numpy as np
import sys
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data_factory import StreamBuilder
from library.model import DualStreamTrainer


def main():
    print("Starting Contact Detection Pipeline Demo...")

    # =========================================================================
    # 1. Configuration Overrides for Speed and Demo Isolation
    # =========================================================================
    print("\n[1] Configuring environment for demo...")

    # Define demo working directory
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load for demo
    Config.STREAM_A_PARAMS["n_estimators"] = 10
    Config.STREAM_A_PARAMS["early_stopping_rounds"] = 5
    Config.STREAM_B_PARAMS["n_estimators"] = 10
    Config.STREAM_B_PARAMS["early_stopping_rounds"] = 5
    Config.THRESHOLD_OPT_STEPS = 10
    Config.NEGATIVE_SAMPLE_RATIO = 2.0  # Keep small for demo balance

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Create Mini-Datasets (Subsetting Metadata)
    # =========================================================================
    print("\n[2] Creating mini-datasets for rapid execution...")

    # Load original metadata
    orig_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    orig_val_meta = pd.read_csv(Config.VAL_META_PATH)

    # Select a small subset of plays (e.g., 2 plays for train, 1 for val)
    # This ensures FeatureGenerator filters the large tracking CSVs effectively
    train_plays = orig_train_meta["game_play"].unique()[:2]
    val_plays = orig_val_meta["game_play"].unique()[:1]

    mini_train_df = orig_train_meta[
        orig_train_meta["game_play"].isin(train_plays)
    ].copy()
    mini_val_df = orig_val_meta[orig_val_meta["game_play"].isin(val_plays)].copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_validation.csv")

    mini_train_df.to_csv(mini_train_path, index=False)
    mini_val_df.to_csv(mini_val_path, index=False)

    print(f"    Mini Train Plays: {len(train_plays)} | Rows: {len(mini_train_df)}")
    print(f"    Mini Val Plays:   {len(val_plays)} | Rows: {len(mini_val_df)}")

    # Point Config to mini metadata
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path

    # =========================================================================
    # 3. Feature Engineering (StreamBuilder)
    # =========================================================================
    print("\n[3] Building Feature Streams...")

    # --- Train Set ---
    print("    Generating Training Data...")
    builder_train = StreamBuilder(mode="train")

    # Build Stream A (Interaction)
    X_train_A, y_train_A, ids_train_A = builder_train.build_interaction_set(
        load_cached=False
    )
    print(f"    Stream A Train Shape: {X_train_A.shape}")

    # Build Stream B (Impact)
    X_train_B, y_train_B, ids_train_B = builder_train.build_impact_set(
        load_cached=False
    )
    print(f"    Stream B Train Shape: {X_train_B.shape}")

    # Assertions
    assert not X_train_A.empty, "Stream A training features are empty."
    assert len(X_train_A) == len(y_train_A), "Stream A features/labels mismatch."
    assert not X_train_B.empty, "Stream B training features are empty."

    # --- Validation Set ---
    print("    Generating Validation Data...")
    builder_val = StreamBuilder(mode="validation")

    # Build Stream A
    X_val_A, y_val_A, ids_val_A = builder_val.build_interaction_set(load_cached=False)

    # Build Stream B
    X_val_B, y_val_B, ids_val_B = builder_val.build_impact_set(load_cached=False)

    print(f"    Stream A Val Shape: {X_val_A.shape}")
    print(f"    Stream B Val Shape: {X_val_B.shape}")

    # =========================================================================
    # 4. Model Training (DualStreamTrainer)
    # =========================================================================
    print("\n[4] Training Dual-Stream Models...")

    trainer = DualStreamTrainer()

    # Fit models
    trainer.fit(
        X_train_A, y_train_A, X_val_A, y_val_A, X_train_B, y_train_B, X_val_B, y_val_B
    )

    # Validate models were created
    assert trainer.model_a is not None, "Stream A model failed to train."
    assert trainer.model_b is not None, "Stream B model failed to train."

    print(f"    Optimized Threshold A: {trainer.threshold_a:.4f}")
    print(f"    Optimized Threshold B: {trainer.threshold_b:.4f}")

    # =========================================================================
    # 5. Inference and Validation
    # =========================================================================
    print("\n[5] Running Inference Validation...")

    # Predict on validation set (Stream A)
    probs_a = trainer.predict(X_val_A, stream_type="A")

    # Basic checks on predictions
    assert len(probs_a) == len(X_val_A), "Prediction length mismatch."
    assert probs_a.min() >= 0.0 and probs_a.max() <= 1.0, "Probabilities out of bounds."

    print(f"    Stream A Predictions (First 5): {probs_a[:5]}")

    # =========================================================================
    # 6. Checkpointing
    # =========================================================================
    print("\n[6] Verifying Checkpoint System...")

    # Save
    trainer.save_checkpoint(base_path=DEMO_DIR)

    # Load into new instance
    new_trainer = DualStreamTrainer()
    new_trainer.load_checkpoint(base_path=DEMO_DIR)

    # Verify state restoration
    assert new_trainer.model_a is not None, "Failed to load Stream A model."
    assert (
        new_trainer.threshold_a == trainer.threshold_a
    ), "Threshold A mismatch after load."
    assert (
        new_trainer.threshold_b == trainer.threshold_b
    ), "Threshold B mismatch after load."

    print("    Checkpoint loaded successfully.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
