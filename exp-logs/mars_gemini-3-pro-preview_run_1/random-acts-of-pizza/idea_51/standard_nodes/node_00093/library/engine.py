import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import MLP_PARAMS, ENSEMBLE_WEIGHTS
from library.utils import seed_everything
from library.mlp_architecture import PizzaNet
from library.rf_learner import predict_rf


def train_mlp_model(train_loader, val_loader, input_metadata_dim, device=None):
    """
    Trains the MLP model with AdamW optimizer and Early Stopping based on Validation AUC.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        input_metadata_dim: Dimension of the metadata features (required for model init).
        device: torch.device (optional).

    Returns:
        model: The trained PizzaNet model with the best validation weights loaded.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    model = PizzaNet(input_metadata_dim=input_metadata_dim)
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MLP_PARAMS["learning_rate"],
        weight_decay=MLP_PARAMS["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Configuration
    max_epochs = MLP_PARAMS["max_epochs"]
    patience = MLP_PARAMS["patience"]

    # Tracking variables for Early Stopping
    best_val_auc = -1.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting MLP training on {device} for {max_epochs} epochs...")

    for epoch in range(1, max_epochs + 1):
        # --- Training Phase ---
        model.train()
        train_losses = []

        for batch in train_loader:
            # Move batch to device
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels = batch_gpu["label"].unsqueeze(1)  # Shape: (B, 1)

            optimizer.zero_grad()
            logits = model(batch_gpu)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                labels = batch_gpu["label"].unsqueeze(1)

                logits = model(batch_gpu)
                probs = torch.sigmoid(logits)

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # Compute Metrics
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_auc = roc_auc_score(val_targets, val_preds)

        # Print full precision metrics
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss}, Val AUC = {val_auc}")

        # --- Early Stopping Logic ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}. Best Val AUC: {best_val_auc}"
            )
            break

    # Load the best weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_mlp(model, data_loader, device=None):
    """
    Generates probability predictions using the trained MLP model.

    Args:
        model: Trained PizzaNet model.
        data_loader: DataLoader containing the data to predict on.
        device: torch.device (optional).

    Returns:
        np.array: Flattened array of probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_gpu)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())

    return np.array(all_preds).flatten()


def evaluate_models(rf_model, mlp_model, test_loader_mlp, X_test_rf, device=None):
    """
    Generates predictions from both the Random Forest and MLP models,
    and combines them using a weighted average.

    Args:
        rf_model: Trained Random Forest model.
        mlp_model: Trained MLP model.
        test_loader_mlp: DataLoader for MLP test data.
        X_test_rf: Sparse matrix or array for RF test data.
        device: torch.device (optional).

    Returns:
        ensemble_preds: np.array of final weighted probabilities.
    """
    # 1. Generate MLP Predictions
    print("Generating MLP predictions...")
    mlp_preds = predict_mlp(mlp_model, test_loader_mlp, device)

    # 2. Generate RF Predictions
    print("Generating RF predictions...")
    rf_preds = predict_rf(rf_model, X_test_rf)

    # 3. Ensemble
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_mlp = ENSEMBLE_WEIGHTS["mlp"]

    print(f"Ensembling with weights: RF={w_rf}, MLP={w_mlp}")
    ensemble_preds = (w_rf * rf_preds) + (w_mlp * mlp_preds)

    return ensemble_preds
