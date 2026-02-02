import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed, suppress_warnings
import library.data_loader as data_loader
import library.feature_engineering as feature_engineering
import library.stream_a_rf as stream_a_rf
import library.stream_b_mlp as stream_b_mlp


def run_demo():
    print("=== Starting Pipeline Demonstration ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment for rapid demonstration...")
    suppress_warnings()
    set_seed(Config.SEED)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use very small subset for demo
    Config.RF_PARAMS["n_estimators"] = 5  # Reduce RF trees
    Config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead in small demo
    Config.NUM_EPOCHS = 1  # Reduce MLP epochs
    Config.TFIDF_MAX_FEATURES = 100  # Reduce TF-IDF vocab
    Config.WORKING_DIR = "./working/demo_execution"  # Separate working dir for demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("\n[2] Loading Data...")
    # Force reload to ensure we use the debug subset
    train_df, val_df, test_df = data_loader.load_data(
        load_cached_data=False, debug=True
    )

    # Validation
    print(f"    Train shape: {train_df.shape}")
    print(f"    Val shape:   {val_df.shape}")
    print(f"    Test shape:  {test_df.shape}")

    assert (
        len(train_df) == Config.DEBUG_SUBSET_SIZE
    ), "Train set size mismatch for debug mode"
    assert (
        len(val_df) == Config.DEBUG_SUBSET_SIZE
    ), "Val set size mismatch for debug mode"
    assert (
        len(test_df) == Config.DEBUG_SUBSET_SIZE
    ), "Test set size mismatch for debug mode"

    # 3. Feature Engineering (Explicit Check)
    print("\n[3] Verifying Feature Processor...")
    fp = feature_engineering.FeatureProcessor()

    # Run processing explicitly to check intermediate outputs
    # Note: In the full pipeline, this is called inside the stream classes,
    # but we call it here to validate the FeatureProcessor class itself.
    data_dicts = fp.process_data(train_df, val_df, test_df, load_cached_data=False)

    # Validate structure
    assert "train" in data_dicts and "val" in data_dicts and "test" in data_dicts
    train_feats = data_dicts["train"]

    # Check specific feature keys
    expected_keys = [
        "X_meta",
        "X_topk",
        "emb_title",
        "emb_body",
        "history_centroids",
        "consistency",
        "history_sequences",
        "y",
    ]
    for k in expected_keys:
        assert k in train_feats, f"Missing key {k} in feature dictionary"

    print(f"    Metadata shape: {train_feats['X_meta'].shape}")
    print(f"    SBERT Title shape: {train_feats['emb_title'].shape}")

    # 4. Stream A: Random Forest Pipeline
    print("\n[4] Running Stream A: Random Forest Pipeline...")
    rf_pipeline = stream_a_rf.RandomForestPipeline()

    # Run pipeline (includes TF-IDF generation and Model Training)
    rf_val_preds, rf_test_preds = rf_pipeline.run(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validate RF outputs
    assert len(rf_val_preds) == len(val_df)
    assert len(rf_test_preds) == len(test_df)
    assert np.all((rf_val_preds >= 0) & (rf_val_preds <= 1)), "RF preds out of bounds"
    print(f"    RF Val Preds Mean: {rf_val_preds.mean():.4f}")

    # 5. Stream B: MLP Pipeline
    print("\n[5] Running Stream B: MLP Pipeline...")
    mlp_pipeline = stream_b_mlp.MLPPipeline()

    # Run pipeline (includes Dataset creation, Training loop, Inference)
    mlp_val_preds, mlp_test_preds = mlp_pipeline.run(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validate MLP outputs
    assert len(mlp_val_preds) == len(val_df)
    assert len(mlp_test_preds) == len(test_df)
    assert np.all(
        (mlp_val_preds >= 0) & (mlp_val_preds <= 1)
    ), "MLP preds out of bounds"
    print(f"    MLP Val Preds Mean: {mlp_val_preds.mean():.4f}")

    # 6. Ensemble and Submission
    print("\n[6] Ensembling and Generating Submission...")

    # Weighted Average
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]

    final_test_preds = (rf_test_preds * w_rf) + (mlp_test_preds * w_mlp)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "request_id": test_df["request_id"],
            "requester_received_pizza": final_test_preds,
        }
    )

    # Validate Submission
    assert len(submission_df) == len(test_df)
    assert submission_df["request_id"].nunique() == len(test_df)

    # Save to working directory (not modifying input)
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"    Submission saved to: {output_path}")
    print(f"    First 5 predictions:\n{submission_df.head()}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
