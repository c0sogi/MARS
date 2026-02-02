import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import time

from library.config import Config
from library.dataset import PizzaDataset
from library.models import DualQueryGatedMLP, RFWrapper


def train_mlp_model(train_features, val_features, device=Config.DEVICE):
    """
    Trains the Dual-Query Gated MLP model with Early Stopping.

    Args:
        train_features (dict): Dictionary of training features for MLP.
        val_features (dict): Dictionary of validation features for MLP.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        model: The trained PyTorch model with best weights loaded.
    """
    print(f"Initializing MLP training on {device}...")

    # Prepare Datasets and DataLoaders
    train_dataset = PizzaDataset(train_features["X_mlp"], labels=train_features["y"])
    val_dataset = PizzaDataset(val_features["X_mlp"], labels=val_features["y"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # Initialize Model
    model = DualQueryGatedMLP(
        embedding_dim=Config.SBERT_EMBEDDING_DIM,
        metadata_dim=len(Config.NUMERIC_COLS),
        hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout_emb=Config.MLP_DROPOUT_EMB,
        dropout_dense=Config.MLP_DROPOUT_DENSE,
    ).to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping Tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = Config.MODEL_MLP_PATH

    print("Starting training loop...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            # Move data to device
            title_emb = batch["title_emb"].to(device)
            body_emb = batch["body_emb"].to(device)
            history_emb = batch["history_emb"].to(device)
            metadata = batch["metadata"].to(device)
            mask = batch["history_padding_mask"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()

            outputs = model(
                title_emb, body_emb, history_emb, metadata, history_padding_mask=mask
            )
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())

        avg_train_loss = train_loss_sum / len(train_dataset)
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
        except ValueError:
            train_auc = 0.5

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                title_emb = batch["title_emb"].to(device)
                body_emb = batch["body_emb"].to(device)
                history_emb = batch["history_emb"].to(device)
                metadata = batch["metadata"].to(device)
                mask = batch["history_padding_mask"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                outputs = model(
                    title_emb,
                    body_emb,
                    history_emb,
                    metadata,
                    history_padding_mask=mask,
                )
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_targets.extend(labels.detach().cpu().numpy())

        avg_val_loss = val_loss_sum / len(val_dataset)
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {avg_train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {avg_val_loss} | Val AUC: {val_auc}"
        )

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(
                f"Early stopping counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def train_rf_model(train_features):
    """
    Trains the Random Forest model.

    Args:
        train_features (dict): Dictionary containing 'X_rf' (sparse matrix) and 'y' (labels).

    Returns:
        rf_wrapper: The trained RFWrapper instance.
    """
    print("Initializing Random Forest training...")
    rf_model = RFWrapper()

    X = train_features["X_rf"]
    y = train_features["y"]

    rf_model.fit(X, y)
    rf_model.save(Config.MODEL_RF_PATH)

    return rf_model


def predict_ensemble(rf_model, mlp_model, test_features, device=Config.DEVICE):
    """
    Generates predictions using the ensemble of RF and MLP.

    Args:
        rf_model (RFWrapper): Trained Random Forest model.
        mlp_model (DualQueryGatedMLP): Trained MLP model.
        test_features (dict): Dictionary containing 'X_rf' and 'X_mlp' for test set.
        device (str): Device for MLP inference.

    Returns:
        final_preds (np.array): Combined probability predictions.
    """
    print("Generating ensemble predictions...")

    # --- RF Predictions ---
    print("Predicting with Random Forest...")
    rf_preds = rf_model.predict_proba(test_features["X_rf"])

    # --- MLP Predictions ---
    print("Predicting with MLP...")
    mlp_model.eval()
    test_dataset = PizzaDataset(test_features["X_mlp"], labels=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    mlp_preds_list = []
    with torch.no_grad():
        for batch in test_loader:
            title_emb = batch["title_emb"].to(device)
            body_emb = batch["body_emb"].to(device)
            history_emb = batch["history_emb"].to(device)
            metadata = batch["metadata"].to(device)
            mask = batch["history_padding_mask"].to(device)

            outputs = mlp_model(
                title_emb, body_emb, history_emb, metadata, history_padding_mask=mask
            )
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            mlp_preds_list.append(probs)

    mlp_preds = np.concatenate(mlp_preds_list)

    # --- Ensemble Averaging ---
    print(
        f"Ensembling with weights: RF={Config.ENSEMBLE_WEIGHT_RF}, MLP={Config.ENSEMBLE_WEIGHT_MLP}"
    )
    final_preds = (Config.ENSEMBLE_WEIGHT_RF * rf_preds) + (
        Config.ENSEMBLE_WEIGHT_MLP * mlp_preds
    )

    return final_preds
