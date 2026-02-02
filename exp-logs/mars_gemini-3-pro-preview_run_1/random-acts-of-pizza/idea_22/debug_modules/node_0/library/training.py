import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data_loader import load_dataset
from library.feature_engineering import FeatureEngineer
from library.neural_net import NeuralNetTrainer


def run_training_pipeline(load_cached_data=True):
    """
    Orchestrates the full training pipeline for the Hybrid Ensemble:
    1. Loads train/val/test datasets.
    2. Generates features using FeatureEngineer (RF and MLP specific).
    3. Trains the Random Forest model (Stream A).
    4. Trains the Gated Attention MLP model (Stream B).
    5. Ensembles predictions via weighted average.
    6. Saves the submission file.

    Args:
        load_cached_data (bool): If True, attempts to load features/data from cache.

    Returns:
        dict: Dictionary containing validation AUC scores for both models.
    """

    # Ensure reproducibility
    np.random.seed(Config.RANDOM_SEED)

    # 1. Load Data
    print("Loading datasets...")
    train_df, val_df, test_df = load_dataset(load_cached_data=load_cached_data)

    # 2. Feature Engineering
    print("Initializing Feature Engineering...")
    fe = FeatureEngineer()

    # Prepare RF Data (Stream A)
    # Includes: Metadata, Target Encoding, Consistency Score, TF-IDF
    print("Preparing Random Forest Inputs...")
    (X_train_rf, y_train_rf), (X_val_rf, y_val_rf), X_test_rf = fe.prepare_rf_inputs(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # Prepare MLP Data (Stream B)
    # Includes: Scaled Metadata, SBERT Request Embeddings, SBERT History Embeddings
    print("Preparing MLP Inputs...")
    train_data_mlp, val_data_mlp, test_data_mlp = fe.prepare_mlp_inputs(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 3. Train Random Forest (Stream A)
    print("Training Random Forest (Stream A)...")
    rf_model = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        class_weight=Config.RF_CLASS_WEIGHT,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        n_jobs=Config.RF_N_JOBS,
        random_state=Config.RANDOM_SEED,
    )

    rf_model.fit(X_train_rf, y_train_rf)

    # Evaluate RF on Validation Set
    rf_val_preds = rf_model.predict_proba(X_val_rf)[:, 1]
    rf_val_auc = roc_auc_score(y_val_rf, rf_val_preds)
    print(f"Random Forest Validation AUC: {rf_val_auc}")

    # Generate RF Test Predictions
    rf_test_preds = rf_model.predict_proba(X_test_rf)[:, 1]

    # 4. Train Gated Attention MLP (Stream B)
    print("Training Gated Attention MLP (Stream B)...")
    # Determine metadata dimension dynamically from the prepared data
    meta_dim = train_data_mlp["metadata"].shape[1]

    trainer = NeuralNetTrainer(input_dims={"metadata": meta_dim})

    # Train the model (returns best validation AUC)
    mlp_val_auc = trainer.train(train_data_mlp, val_data_mlp)
    print(f"MLP Best Validation AUC: {mlp_val_auc}")

    # Generate MLP Test Predictions
    mlp_test_preds = trainer.predict(test_data_mlp)

    # 5. Ensemble Predictions
    print("Ensembling predictions...")
    # Simple Weighted Average (0.5 / 0.5)
    if rf_test_preds.shape != mlp_test_preds.shape:
        raise ValueError(
            f"Shape mismatch in predictions: RF {rf_test_preds.shape}, MLP {mlp_test_preds.shape}"
        )

    final_test_preds = 0.5 * rf_test_preds + 0.5 * mlp_test_preds

    # 6. Generate Submission File
    print("Generating submission file...")
    submission_df = pd.DataFrame(
        {
            "request_id": test_df["request_id"],
            "requester_received_pizza": final_test_preds,
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return {"rf_val_auc": rf_val_auc, "mlp_val_auc": mlp_val_auc}
