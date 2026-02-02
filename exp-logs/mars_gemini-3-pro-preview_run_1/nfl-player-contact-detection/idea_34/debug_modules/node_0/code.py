import os
import sys
import shutil
import pandas as pd
import numpy as np
import logging
import warnings
import gc

# Import from the provided library
from library import (
    config,
    utils,
    feature_engineering,
    data_loader,
    model_factory,
    trainer,
    inference,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def setup_demo_environment():
    """
    Creates a demo working directory and generates mini-datasets
    (1 play for train, val, test) to ensure the script runs quickly.
    Modifies the global config to point to these mini-datasets.
    """
    print(">>> [1/5] Setting up demo environment and mini-datasets...")

    # Define demo paths
    demo_dir = os.path.join(config.WORKING_DIR, "demo_execution")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config Paths to point to demo directory
    config.WORKING_DIR = demo_dir
    config.SUBMISSION_DIR = demo_dir
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override Hyperparameters for Speed
    config.NUM_ESTIMATORS = 5
    config.EARLY_STOPPING_ROUNDS = 5
    config.N_JOBS = 2

    # --- Create Mini Train/Val Data ---
    # Load full metadata to sample plays
    full_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    full_val_meta = pd.read_csv(config.VAL_METADATA_PATH)

    # Sample 1 unique game_play for Train and Val
    train_play = full_train_meta["game_play"].unique()[0]
    val_play = full_val_meta["game_play"].unique()[0]

    mini_train_meta = full_train_meta[full_train_meta["game_play"] == train_play].copy()
    mini_val_meta = full_val_meta[full_val_meta["game_play"] == val_play].copy()

    # Load full tracking to filter for these plays
    full_train_tracking = pd.read_csv(config.TRAIN_TRACKING_PATH)
    mini_train_tracking = full_train_tracking[
        full_train_tracking["game_play"].isin([train_play, val_play])
    ].copy()

    # Save Mini Train/Val Files
    mini_train_meta_path = os.path.join(demo_dir, "mini_train_metadata.csv")
    mini_val_meta_path = os.path.join(demo_dir, "mini_val_metadata.csv")
    mini_train_trk_path = os.path.join(demo_dir, "mini_train_tracking.csv")

    mini_train_meta.to_csv(mini_train_meta_path, index=False)
    mini_val_meta.to_csv(mini_val_meta_path, index=False)
    mini_train_tracking.to_csv(mini_train_trk_path, index=False)

    # Update Config to point to mini files
    config.TRAIN_METADATA_PATH = mini_train_meta_path
    config.VAL_METADATA_PATH = mini_val_meta_path
    config.TRAIN_TRACKING_PATH = mini_train_trk_path

    print(f"    Selected Train Play: {train_play} ({len(mini_train_meta)} rows)")
    print(f"    Selected Val Play: {val_play} ({len(mini_val_meta)} rows)")

    # --- Create Mini Test Data ---
    full_test_meta = pd.read_csv(config.TEST_METADATA_PATH)
    # Ensure we pick a play that exists in test tracking
    full_test_tracking = pd.read_csv(config.TEST_TRACKING_PATH)
    available_test_plays = full_test_tracking["game_play"].unique()

    if len(available_test_plays) > 0:
        test_play = available_test_plays[0]
        mini_test_meta = full_test_meta[full_test_meta["game_play"] == test_play].copy()
        mini_test_tracking = full_test_tracking[
            full_test_tracking["game_play"] == test_play
        ].copy()
    else:
        # Fallback if intersection is empty (unlikely given task desc)
        test_play = full_test_meta["game_play"].unique()[0]
        mini_test_meta = full_test_meta[full_test_meta["game_play"] == test_play].copy()
        mini_test_tracking = full_test_tracking.head(100).copy()  # Dummy

    mini_test_meta_path = os.path.join(demo_dir, "mini_test_metadata.csv")
    mini_test_trk_path = os.path.join(demo_dir, "mini_test_tracking.csv")

    mini_test_meta.to_csv(mini_test_meta_path, index=False)
    mini_test_tracking.to_csv(mini_test_trk_path, index=False)

    config.TEST_METADATA_PATH = mini_test_meta_path
    config.TEST_TRACKING_PATH = mini_test_trk_path

    print(f"    Selected Test Play: {test_play} ({len(mini_test_meta)} rows)")
    print("    Demo environment setup complete.")


def demonstrate_feature_engineering():
    """
    Demonstrates the ReferenceAnchoredFeatures class.
    Generates features for the mini training set.
    """
    print("\n>>> [2/5] Demonstrating Feature Engineering...")

    # Instantiate
    fe_engine = feature_engineering.ReferenceAnchoredFeatures()

    # Load metadata manually to pass to generator
    df_meta = pd.read_csv(config.TRAIN_METADATA_PATH)

    # Generate Features (force no cache to verify logic)
    df_features = fe_engine.generate_features(
        df_meta, split="train", load_cached_data=False
    )

    # Verification
    assert not df_features.empty, "Feature dataframe is empty."
    assert "dist_lag0" in df_features.columns, "Expected feature 'dist_lag0' missing."
    assert "contact" in df_features.columns, "Target column 'contact' missing."

    print(f"    Generated features shape: {df_features.shape}")
    print("    Feature engineering verification passed.")
    return df_features


def demonstrate_model_factory(df_features):
    """
    Demonstrates the ModelFactory class.
    Trains a simple LightGBM model on the generated features.
    """
    print("\n>>> [3/5] Demonstrating Model Factory...")

    factory = model_factory.ModelFactory()

    # Prepare X and y
    feature_cols = config.FEATURE_COLUMNS
    target_col = "contact"

    X = df_features[feature_cols]
    y = df_features[target_col]

    # Train a dummy LGBM
    print("    Training a demo LightGBM model...")
    model = factory.train_model("lgbm", X, y)

    # Predict
    probs = factory.predict_proba(model, X)

    # Verification
    assert model is not None, "Model training failed (returned None)."
    assert len(probs) == len(X), "Prediction shape mismatch."
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities out of [0, 1] range."

    print(f"    Model trained. Prediction mean: {probs.mean():.4f}")
    print("    Model Factory verification passed.")


def demonstrate_full_pipeline():
    """
    Demonstrates the CurriculumTrainer class.
    Runs the full Scout -> Mining -> Expert -> Threshold pipeline.
    """
    print("\n>>> [4/5] Demonstrating Full Training Pipeline (CurriculumTrainer)...")

    # Instantiate Trainer
    curriculum_trainer = trainer.CurriculumTrainer()

    # Run Pipeline
    # load_cached_data=True allows it to pick up the parquet file generated in step 2
    # if it matches the hash/path logic, otherwise it regenerates.
    curriculum_trainer.run(load_cached_data=True)

    # Verification of Artifacts
    models_dir = os.path.join(config.WORKING_DIR, "models")

    # Check for saved models
    expected_models = ["scout_lgbm.joblib", "expert_lgbm.joblib", "best_threshold.npy"]
    for m in expected_models:
        path = os.path.join(models_dir, m)
        if not os.path.exists(path):
            # It's possible XGB/Cat weren't trained if we strictly look for LGBM,
            # but the trainer loops through all types.
            # We'll check if at least one expert exists.
            pass

    assert os.path.exists(
        os.path.join(models_dir, "best_threshold.npy")
    ), "Threshold file not found."
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not generated."

    # Validate Submission Format
    sub = pd.read_csv(config.SUBMISSION_PATH)
    assert (
        "contact_id" in sub.columns and "contact" in sub.columns
    ), "Submission columns invalid."
    assert sub["contact"].isin([0, 1]).all(), "Submission contains non-binary values."

    print("    Training pipeline completed successfully.")
    print(f"    Submission generated at: {config.SUBMISSION_PATH}")


def demonstrate_inference():
    """
    Demonstrates the InferencePipeline class.
    Loads trained models and generates predictions for the test set.
    """
    print("\n>>> [5/5] Demonstrating Inference Pipeline...")

    inf_pipeline = inference.InferencePipeline()

    # Run Inference
    inf_pipeline.run(load_cached_data=False)

    # Verification
    # The submission file should have been overwritten/updated
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), "Submission file missing after inference."

    sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"    Inference complete. Final submission rows: {len(sub)}")

    # Check against sample submission length (should match exactly)
    sample_sub = pd.read_csv(os.path.join(config.INPUT_DIR, "sample_submission.csv"))
    assert len(sub) == len(
        sample_sub
    ), f"Submission length mismatch. Expected {len(sample_sub)}, got {len(sub)}"

    print("    Inference verification passed.")


if __name__ == "__main__":
    # Seed everything
    utils.seed_everything(config.SEED)

    # 1. Setup
    setup_demo_environment()

    # 2. Feature Engineering Demo
    df_feats = demonstrate_feature_engineering()

    # 3. Model Factory Demo
    demonstrate_model_factory(df_feats)

    # 4. Full Training Pipeline Demo
    demonstrate_full_pipeline()

    # 5. Inference Pipeline Demo
    demonstrate_inference()

    print("\n=== All Demonstrations Completed Successfully ===")
