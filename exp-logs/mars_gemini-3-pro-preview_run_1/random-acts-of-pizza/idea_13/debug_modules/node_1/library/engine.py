import numpy as np
import pandas as pd
from library.config import Config
from library.data_utils import load_data, save_submission
from library.text_utils import SBERTEmbedder
from library.topic_utils import train_topic_model_if_needed
from library.feature_builder import FeatureBuilder
from library.rf_model import TopicAlignedRF
from library.mlp_model import MLPTrainer


def train_rf_stream(
    train_df, val_df, embedder, aligner, feature_builder, load_cached_data=True
):
    """
    Orchestrates the training of the Topic-Aligned Random Forest (Stream A).
    """
    print("Preparing data for Random Forest Stream...")
    # Prepare features: Tabular + Topic Alignment + TF-IDF
    X_train, y_train = feature_builder.prepare_rf_inputs(
        train_df, "train", embedder, aligner, load_cached_data=load_cached_data
    )
    X_val, y_val = feature_builder.prepare_rf_inputs(
        val_df, "val", embedder, aligner, load_cached_data=load_cached_data
    )

    # Initialize and Train Model
    rf_model = TopicAlignedRF()
    rf_model.train(X_train, y_train)

    # Evaluate
    val_auc = rf_model.evaluate(X_val, y_val)

    return rf_model, val_auc


def train_mlp_stream(
    train_df, val_df, embedder, feature_builder, load_cached_data=True
):
    """
    Orchestrates the training of the Credibility-Gated MLP (Stream B).
    """
    print("Preparing data for MLP Stream...")
    # Prepare tensors: SBERT Embeddings + Scaled Metadata
    train_data = feature_builder.prepare_mlp_inputs(
        train_df, "train", embedder, load_cached_data=load_cached_data
    )
    val_data = feature_builder.prepare_mlp_inputs(
        val_df, "val", embedder, load_cached_data=load_cached_data
    )

    # Initialize Trainer
    # Input dim comes from SBERT (384), Meta dim from feature builder (13 engineered features)
    mlp_trainer = MLPTrainer()

    # Train with Early Stopping
    val_auc = mlp_trainer.train(train_data, val_data)

    return mlp_trainer, val_auc


def generate_ensemble_predictions(
    test_df,
    rf_model,
    mlp_trainer,
    embedder,
    aligner,
    feature_builder,
    load_cached_data=True,
):
    """
    Generates predictions from both streams, averages them, and saves the submission.
    """
    print("Generating predictions for Test Set...")

    # --- Random Forest Inference ---
    X_test, _ = feature_builder.prepare_rf_inputs(
        test_df, "test", embedder, aligner, load_cached_data=load_cached_data
    )
    rf_probs = rf_model.predict_proba(X_test)
    print(f"RF Predictions Generated. Mean Prob: {np.mean(rf_probs):.4f}")

    # --- MLP Inference ---
    test_data_mlp = feature_builder.prepare_mlp_inputs(
        test_df, "test", embedder, load_cached_data=load_cached_data
    )
    mlp_probs = mlp_trainer.predict_proba(test_data_mlp)
    print(f"MLP Predictions Generated. Mean Prob: {np.mean(mlp_probs):.4f}")

    # --- Weighted Ensemble ---
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]

    print(f"Ensembling with weights -> RF: {w_rf}, MLP: {w_mlp}")
    final_probs = (w_rf * rf_probs) + (w_mlp * mlp_probs)

    # --- Save Submission ---
    request_ids = test_df["request_id"].values
    save_submission(request_ids, final_probs)

    return final_probs


def run_full_pipeline(load_cached_data=True):
    """
    Main entry point for the training and inference pipeline.
    """
    # 1. Load Data
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # 2. Initialize Shared Components
    # SBERTEmbedder handles text embeddings for both streams
    embedder = SBERTEmbedder()

    # FeatureBuilder handles stateful transformations (TF-IDF, Scalers)
    feature_builder = FeatureBuilder()

    # 3. Topic Modeling (Pre-requisite for RF Stream)
    # Fits NMF on training requests
    aligner = train_topic_model_if_needed(
        train_df, embedder, load_cached_data=load_cached_data
    )

    # 4. Stream A: Random Forest
    print("\n" + "=" * 40)
    print("STREAM A: Topic-Aligned Random Forest")
    print("=" * 40)
    rf_model, rf_auc = train_rf_stream(
        train_df,
        val_df,
        embedder,
        aligner,
        feature_builder,
        load_cached_data=load_cached_data,
    )

    # 5. Stream B: MLP
    print("\n" + "=" * 40)
    print("STREAM B: Credibility-Gated MLP")
    print("=" * 40)
    mlp_trainer, mlp_auc = train_mlp_stream(
        train_df, val_df, embedder, feature_builder, load_cached_data=load_cached_data
    )

    # 6. Summary
    print("\n" + "=" * 40)
    print("TRAINING COMPLETE")
    print(f"Random Forest Validation AUC: {rf_auc}")
    print(f"MLP Validation AUC:           {mlp_auc}")
    print("=" * 40 + "\n")

    # 7. Inference & Submission
    generate_ensemble_predictions(
        test_df,
        rf_model,
        mlp_trainer,
        embedder,
        aligner,
        feature_builder,
        load_cached_data=load_cached_data,
    )
