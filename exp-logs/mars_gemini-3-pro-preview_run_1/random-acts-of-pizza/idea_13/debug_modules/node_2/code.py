import sys
import os
import numpy as np
import pandas as pd
import torch

# Import from the provided library
from library.config import Config
from library.data_utils import load_data
from library.text_utils import SBERTEmbedder
from library.topic_utils import train_topic_model_if_needed, TopicAligner
from library.feature_builder import FeatureBuilder
from library.rf_model import TopicAlignedRF
from library.mlp_model import MLPTrainer
from library.engine import generate_ensemble_predictions


def run_demo():
    print("=== Starting Pizza Request Prediction Demo ===")

    # 1. Configuration Overrides for Speed and Isolation
    # We modify the Config class attributes directly to create a sandbox for this demo
    print("1. Configuring environment...")
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_output"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce hyperparameters for rapid execution
    Config.RF_PARAMS["n_estimators"] = 10  # Reduced from 500
    Config.RF_PARAMS["n_jobs"] = 1  # Sequential for demo stability

    Config.MLP_PARAMS["epochs"] = 2  # Reduced from 50
    Config.MLP_PARAMS["batch_size"] = 8  # Small batch for small subset
    Config.MLP_PARAMS["hidden_dim"] = 32  # Smaller network
    Config.MLP_PARAMS["patience"] = 2  # Strict early stopping

    # 2. Data Loading & Subsampling
    print("2. Loading and subsampling data...")
    # Load raw data (skipping existing cache to ensure logic runs)
    train_df, val_df, test_df = load_data(load_cached_data=False)

    # Create small subsets
    train_subset = train_df.iloc[:50].copy().reset_index(drop=True)
    val_subset = val_df.iloc[:20].copy().reset_index(drop=True)
    test_subset = test_df.iloc[:20].copy().reset_index(drop=True)

    print(
        f"   Subset sizes -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # 3. Component Initialization
    print("3. Initializing core components...")
    embedder = SBERTEmbedder()
    feature_builder = FeatureBuilder()

    # 4. Topic Modeling (Pre-requisite for RF Stream)
    print("4. Training Topic Model...")
    # Fits NMF on the training subset requests
    aligner = train_topic_model_if_needed(
        train_subset, embedder, load_cached_data=False
    )

    # Validation: Check if aligner is fitted and functional
    assert isinstance(aligner, TopicAligner)
    assert aligner.model is not None, "TopicAligner should be fitted."
    dummy_emb = np.random.rand(5, 384)  # 5 samples, 384 dim
    dummy_topics = aligner.transform(dummy_emb)
    assert dummy_topics.shape == (
        5,
        Config.NUM_TOPICS,
    ), f"Topic shape mismatch: {dummy_topics.shape}"
    print("   Topic Model verified.")

    # 5. Stream A: Random Forest Pipeline
    print("5. Running Stream A (Random Forest)...")

    # Prepare Inputs (Tabular + Topic + TF-IDF)
    print("   Preparing RF inputs...")
    X_train_rf, y_train_rf = feature_builder.prepare_rf_inputs(
        train_subset, "train", embedder, aligner, load_cached_data=False
    )
    X_val_rf, y_val_rf = feature_builder.prepare_rf_inputs(
        val_subset, "val", embedder, aligner, load_cached_data=False
    )

    # Validation: Check input shapes
    print(f"   RF Input Shape: {X_train_rf.shape}")
    assert X_train_rf.shape[0] == 50
    assert y_train_rf.shape[0] == 50
    assert X_val_rf.shape[0] == 20
    # Ensure feature consistency
    assert (
        X_train_rf.shape[1] == X_val_rf.shape[1]
    ), "Feature dimension mismatch between train and val"

    # Train RF
    print("   Training RF...")
    rf_model = TopicAlignedRF(params=Config.RF_PARAMS)
    rf_model.train(X_train_rf, y_train_rf)

    # Evaluate RF
    rf_auc = rf_model.evaluate(X_val_rf, y_val_rf)
    print(f"   RF Demo AUC: {rf_auc:.4f}")
    assert 0 <= rf_auc <= 1, "AUC must be between 0 and 1"

    # 6. Stream B: MLP Pipeline
    print("6. Running Stream B (MLP)...")

    # Prepare Inputs (Embeddings + Metadata Tensors)
    print("   Preparing MLP inputs...")
    train_data_mlp = feature_builder.prepare_mlp_inputs(
        train_subset, "train", embedder, load_cached_data=False
    )
    val_data_mlp = feature_builder.prepare_mlp_inputs(
        val_subset, "val", embedder, load_cached_data=False
    )

    # Validation: Check dictionary structure and tensor shapes
    required_keys = ["request_emb", "history_emb", "meta_features", "y"]
    for key in required_keys:
        assert key in train_data_mlp, f"Missing key {key} in MLP data"

    assert train_data_mlp["request_emb"].shape == (50, 384)
    assert train_data_mlp["history_emb"].shape == (50, Config.MAX_HISTORY_LENGTH, 384)
    # Meta features: 9 raw + 1 ratio + 1 len + 1 caps + 3 lex = 15 features
    assert train_data_mlp["meta_features"].shape[1] == 15

    # Train MLP
    print("   Training MLP...")
    mlp_trainer = MLPTrainer(params=Config.MLP_PARAMS)
    mlp_auc = mlp_trainer.train(train_data_mlp, val_data_mlp)
    print(f"   MLP Demo AUC: {mlp_auc:.4f}")
    assert 0 <= mlp_auc <= 1

    # 7. Ensemble Prediction on Test
    print("7. Generating Test Predictions...")

    final_probs = generate_ensemble_predictions(
        test_subset,
        rf_model,
        mlp_trainer,
        embedder,
        aligner,
        feature_builder,
        load_cached_data=False,
    )

    # Validation: Check predictions and submission file
    assert len(final_probs) == 20
    assert np.all(
        (final_probs >= 0) & (final_probs <= 1)
    ), "Probabilities must be in [0, 1]"

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(submission_df) == 20
    assert "request_id" in submission_df.columns
    assert "requester_received_pizza" in submission_df.columns

    print(f"   Submission saved to: {Config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
