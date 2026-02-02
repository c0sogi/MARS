import os
import pandas as pd
import numpy as np
from library.config import WORKING_DIR, SUBMISSION_DIR, SEED, ID_COL, TARGET_COL
from library.data_loader import load_dataset
from library.text_processing import generate_sbert_embeddings, generate_tfidf_features
from library.feature_engineering import prepare_rf_features, prepare_mlp_features
from library.rf_model import InteractionRandomForest
from library.mlp_model import MLPTrainer


def train_rf_model(rf_features):
    """
    Trains the Random Forest model using the InteractionRandomForest class.

    Args:
        rf_features (dict): Dictionary containing RF feature matrices and targets.

    Returns:
        InteractionRandomForest: The trained model instance.
    """
    model = InteractionRandomForest()

    # Extract data
    X_train = rf_features["X_train"]
    y_train = rf_features["y_train"]
    X_val = rf_features.get("X_val")
    y_val = rf_features.get("y_val")

    # Train
    model.train(X_train, y_train, X_val, y_val)

    # Save
    model.save()

    return model


def train_mlp_model(mlp_features):
    """
    Trains the MLP model using the MLPTrainer class.

    Args:
        mlp_features (dict): Dictionary containing MLP input tensors/arrays.

    Returns:
        MLPTrainer: The trained trainer instance.
    """
    trainer = MLPTrainer()

    # Train (handles validation loop, optimizer, and early stopping internally)
    trainer.train(mlp_features)

    return trainer


def generate_submission(test_df, rf_preds, mlp_preds, ensemble_weights=(0.5, 0.5)):
    """
    Generates the submission file by ensembling predictions.

    Args:
        test_df (pd.DataFrame): Test dataframe containing request_ids.
        rf_preds (np.ndarray): Predictions from Random Forest.
        mlp_preds (np.ndarray): Predictions from MLP.
        ensemble_weights (tuple): Weights for (RF, MLP).
    """
    # Ensemble
    w_rf, w_mlp = ensemble_weights
    final_preds = (w_rf * rf_preds) + (w_mlp * mlp_preds)

    # Create DataFrame
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: final_preds})

    # Save
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return submission


def run(debug_size=None, load_cached_data=True, train=True):
    """
    Main execution function for the pipeline.

    Args:
        debug_size (int, optional): Number of samples to use for debugging.
        load_cached_data (bool): Whether to load features from cache.
        train (bool): Whether to retrain models or load from disk.
    """
    print("Starting pipeline...")

    # If debugging, disable cache loading to prevent shape mismatches between
    # subsampled data and full cached features.
    if debug_size is not None:
        print(
            "Debug mode enabled: Disabling cache loading to prevent shape mismatches."
        )
        load_cached_data = False

    # 1. Load Data
    train_df, val_df, test_df = load_dataset(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    # 2. Generate Base Text Features
    # These functions handle their own caching logic internally
    sbert_data = generate_sbert_embeddings(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    tfidf_data = generate_tfidf_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 3. Prepare Model-Specific Features
    rf_features = prepare_rf_features(
        train_df,
        val_df,
        test_df,
        tfidf_data,
        sbert_data,
        load_cached_data=load_cached_data,
    )

    mlp_features = prepare_mlp_features(
        train_df, val_df, test_df, sbert_data, load_cached_data=load_cached_data
    )

    # 4. Train Models
    if train:
        print("\n=== Training Random Forest (Stream A) ===")
        rf_model = train_rf_model(rf_features)

        print("\n=== Training MLP (Stream B) ===")
        mlp_trainer = train_mlp_model(mlp_features)
    else:
        print("\n=== Loading Pre-trained Models ===")
        rf_model = InteractionRandomForest()
        if not rf_model.load():
            print("RF Model not found, training...")
            rf_model = train_rf_model(rf_features)

        mlp_trainer = MLPTrainer()
        if not mlp_trainer.load():
            print("MLP Model not found, training...")
            mlp_trainer = train_mlp_model(mlp_features)

    # 5. Inference
    print("\n=== Running Inference ===")

    # RF Inference
    X_test_rf = rf_features["X_test"]
    rf_preds = rf_model.predict_proba(X_test_rf)

    # MLP Inference
    mlp_preds = mlp_trainer.predict_proba(mlp_features, split_name="test")

    # 6. Submission
    generate_submission(test_df, rf_preds, mlp_preds)

    print("Pipeline completed successfully.")
