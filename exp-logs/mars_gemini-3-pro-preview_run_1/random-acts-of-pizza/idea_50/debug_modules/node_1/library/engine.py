import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything


def train_mlp(model, train_loader, val_loader, device):
    """
    Trains the FiLM Classifier MLP with Early Stopping.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run training on.

    Returns:
        model (nn.Module): The model with the best validation weights loaded.
        float: The best validation AUC achieved.
    """
    # Setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.MLP_LEARNING_RATE,
        weight_decay=Config.MLP_WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_mlp_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting MLP training on {device}...")

    for epoch in range(Config.MLP_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch_inputs, batch_labels in train_loader:
            # Move data to device
            # batch_inputs is a dict of tensors
            inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            labels = batch_labels.to(device).unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        val_auc, val_loss = evaluate_mlp(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # --- Early Stopping Check ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.MLP_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc:.10f}"
                )
                break

    # Load best weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model weights.")

    return model, best_auc


def evaluate_mlp(model, data_loader, device, criterion=None):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader containing the dataset.
        device (torch.device): Device to run evaluation on.
        criterion (nn.Module, optional): Loss function.

    Returns:
        float: ROC-AUC score.
        float: Average loss (if criterion provided, else 0.0).
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            labels = batch_labels.to(device).unsqueeze(1)

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            if criterion:
                loss = criterion(logits, labels)
                total_loss += loss.item()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    # Handle edge case where only one class is present in batch/loader (rare but possible in debug)
    if len(np.unique(all_labels)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    avg_loss = total_loss / len(data_loader) if criterion else 0.0

    return auc, avg_loss


def train_rf(model, X_train, y_train):
    """
    Trains the Random Forest model.

    Args:
        model (RandomForestModel): Wrapper class for RF.
        X_train (sparse matrix): Training features.
        y_train (array): Training labels.
    """
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    print("Random Forest training complete.")


def predict_ensemble(
    rf_model, mlp_model, rf_test_feats, mlp_test_loader, test_df, device
):
    """
    Generates predictions using the ensemble of RF and MLP.
    Saves the submission file.

    Args:
        rf_model (RandomForestModel): Trained RF model.
        mlp_model (nn.Module): Trained MLP model.
        rf_test_feats (sparse matrix): Features for RF.
        mlp_test_loader (DataLoader): DataLoader for MLP test data.
        test_df (pd.DataFrame): Test dataframe containing request_ids.
        device (torch.device): Device for MLP inference.
    """
    print("Generating ensemble predictions...")

    # 1. Random Forest Predictions
    rf_probs = rf_model.predict_proba(rf_test_feats)

    # 2. MLP Predictions
    mlp_model.eval()
    mlp_probs_list = []
    with torch.no_grad():
        # Iterate over loader (no labels expected in test loader usually,
        # but PizzaDataset might return just inputs if labels are None)
        for batch in mlp_test_loader:
            # Check if batch is tuple (inputs, labels) or just inputs
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = mlp_model(inputs)
            probs = torch.sigmoid(logits)
            mlp_probs_list.extend(probs.cpu().numpy())

    mlp_probs = np.array(mlp_probs_list).flatten()

    # Ensure lengths match
    if len(rf_probs) != len(mlp_probs):
        raise ValueError(
            f"Shape mismatch: RF preds {len(rf_probs)}, MLP preds {len(mlp_probs)}"
        )

    # 3. Weighted Average
    w_rf = Config.ENSEMBLE_WEIGHT_RF
    w_mlp = Config.ENSEMBLE_WEIGHT_MLP

    final_probs = (w_rf * rf_probs) + (w_mlp * mlp_probs)

    # 4. Create Submission
    submission = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": final_probs}
    )

    # Save
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(submission.head())
