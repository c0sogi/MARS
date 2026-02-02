import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    MLP_BATCH_SIZE,
    ENSEMBLE_WEIGHTS,
    SUBMISSION_FILE,
    SUBMISSION_DIR,
    WORKING_DIR,
)
from library.dataset import PizzaDataset
from library.mlp_model import train_mlp
from library.rf_model import InteractionRandomForest
from library.utils import set_seed


def train_models(
    train_df,
    val_df,
    sbert_train,
    sbert_val,
    tabular_train,
    tabular_val,
    save_models=True,
):
    """
    Orchestrates the training of both the MLP and Random Forest models.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        sbert_train (dict): SBERT features for training.
        sbert_val (dict): SBERT features for validation.
        tabular_train (dict): Tabular features for training.
        tabular_val (dict): Tabular features for validation.
        save_models (bool): Whether to save the trained models to disk.

    Returns:
        tuple: (mlp_model, rf_model)
    """
    # 1. Train MLP
    print("\n=== Training Orthogonal Skip-Gated MLP ===")

    # Prepare Datasets
    # Extract labels
    y_train = train_df["requester_received_pizza"].values
    y_val = val_df["requester_received_pizza"].values

    train_dataset = PizzaDataset(sbert_train, tabular_train, labels=y_train)
    val_dataset = PizzaDataset(sbert_val, tabular_val, labels=y_val)

    # Determine metadata dimension for the MLP input
    metadata_dim = tabular_train["mlp_metadata"].shape[1]

    # Define save path
    mlp_save_path = os.path.join(WORKING_DIR, "best_mlp.pth") if save_models else None

    # Train
    mlp_model = train_mlp(
        train_dataset, val_dataset, metadata_dim, save_path=mlp_save_path
    )

    # 2. Train Random Forest
    print("\n=== Training Interaction-Enhanced Random Forest ===")

    rf_model = InteractionRandomForest()
    rf_save_path = os.path.join(WORKING_DIR, "rf_model.joblib") if save_models else None

    rf_model.train(
        train_features=tabular_train,
        train_labels=y_train,
        val_features=tabular_val,
        val_labels=y_val,
        save_path=rf_save_path,
    )

    return mlp_model, rf_model


def get_mlp_predictions(model, dataset):
    """
    Generates probability predictions using the trained MLP model.

    Args:
        model (nn.Module): Trained MLP model.
        dataset (PizzaDataset): Dataset to predict on.

    Returns:
        np.ndarray: Array of probabilities (0-1).
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=MLP_BATCH_SIZE, shuffle=False, num_workers=0
    )

    all_probs = []

    print("Generating MLP predictions...")
    with torch.no_grad():
        for batch in loader:
            # Move inputs to device
            title = batch["title_emb"].to(DEVICE)
            body = batch["body_emb"].to(DEVICE)
            hist_seq = batch["hist_seq"].to(DEVICE)
            hist_mask = batch["hist_mask"].to(DEVICE)
            centroid = batch["hist_centroid"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)

            # Forward pass
            logits = model(title, body, hist_seq, hist_mask, centroid, meta)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    # Concatenate and flatten
    return np.vstack(all_probs).flatten()


def predict_ensemble(mlp_model, rf_model, test_df, sbert_test, tabular_test):
    """
    Generates final ensemble predictions for the test set.

    Args:
        mlp_model: Trained MLP model.
        rf_model: Trained Random Forest model wrapper.
        test_df (pd.DataFrame): Test metadata.
        sbert_test (dict): SBERT features for test set.
        tabular_test (dict): Tabular features for test set.

    Returns:
        np.ndarray: Final ensemble probabilities.
    """
    print("\n=== Generating Ensemble Predictions ===")

    # 1. MLP Predictions
    test_dataset = PizzaDataset(sbert_test, tabular_test, labels=None)
    mlp_probs = get_mlp_predictions(mlp_model, test_dataset)

    # 2. Random Forest Predictions
    # rf_model.predict_proba expects the feature dictionary
    rf_probs = rf_model.predict_proba(tabular_test)

    # 3. Weighted Ensemble
    # ENSEMBLE_WEIGHTS = [RF_Weight, MLP_Weight]
    w_rf, w_mlp = ENSEMBLE_WEIGHTS

    # Normalize weights
    total_weight = w_rf + w_mlp
    w_rf /= total_weight
    w_mlp /= total_weight

    print(f"Ensembling with weights -> RF: {w_rf:.2f}, MLP: {w_mlp:.2f}")

    final_probs = (w_rf * rf_probs) + (w_mlp * mlp_probs)

    return final_probs


def save_submission(test_df, predictions):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        test_df (pd.DataFrame): Test DataFrame containing 'request_id'.
        predictions (np.ndarray): Array of predicted probabilities.
    """
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    submission = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": predictions}
    )

    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")
