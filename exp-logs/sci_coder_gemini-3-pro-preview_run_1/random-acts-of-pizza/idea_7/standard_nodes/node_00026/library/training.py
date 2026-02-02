import copy
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import RF_PARAMS, MLP_PARAMS, TRAIN_PARAMS, RANDOM_STATE
from library.models import DualBranchMLP
from library.utils import set_seed, save_submission


def train_rf(X_tab, X_tfidf, y):
    """
    Trains the Augmented Random Forest model.

    Args:
        X_tab (np.ndarray): Dense tabular features (raw).
        X_tfidf (scipy.sparse.csr_matrix): Sparse TF-IDF features.
        y (np.ndarray): Target labels.

    Returns:
        RandomForestClassifier: Fitted model.
    """
    print("Initializing Random Forest training...")
    # Combine sparse text features with dense tabular features
    # Convert tabular to sparse to allow efficient hstack
    X_combined = sp.hstack([X_tfidf, sp.csr_matrix(X_tab)])

    print(f"RF Input Shape: {X_combined.shape}")

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_combined, y)

    return rf


def predict_rf(model, X_tab, X_tfidf):
    """
    Generates probabilities using the trained Random Forest.
    """
    X_combined = sp.hstack([X_tfidf, sp.csr_matrix(X_tab)])
    # Return probability of the positive class (index 1)
    return model.predict_proba(X_combined)[:, 1]


def train_mlp(X_train_tab, X_train_sbert, y_train, X_val_tab, X_val_sbert, y_val):
    """
    Trains the Domain-Aware Dual-Branch MLP with Early Stopping.
    """
    print("Initializing MLP training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Convert data to tensors
    t_X_train_tab = torch.tensor(X_train_tab, dtype=torch.float32)
    t_X_train_sbert = torch.tensor(X_train_sbert, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    t_X_val_tab = torch.tensor(X_val_tab, dtype=torch.float32).to(device)
    t_X_val_sbert = torch.tensor(X_val_sbert, dtype=torch.float32).to(device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # Create DataLoader
    dataset = TensorDataset(t_X_train_tab, t_X_train_sbert, t_y_train)
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=TRAIN_PARAMS["num_workers"],
    )

    # Initialize Model
    # meta_dim is derived from the input tabular data shape
    meta_dim = X_train_tab.shape[1]
    model = DualBranchMLP(meta_dim=meta_dim).to(device)

    # Optimizer and Loss
    # Using BCELoss because model returns torch.sigmoid()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MLP_PARAMS["learning_rate"],
        weight_decay=MLP_PARAMS["weight_decay"],
    )
    criterion = nn.BCELoss()

    # Training Loop Variables
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(TRAIN_PARAMS["epochs"]):
        model.train()
        running_loss = 0.0

        for batch_tab, batch_sbert, batch_y in loader:
            batch_tab = batch_tab.to(device)
            batch_sbert = batch_sbert.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            preds = model(batch_sbert, batch_tab)
            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_y.size(0)

        epoch_loss = running_loss / len(dataset)

        # Validation Step
        model.eval()
        with torch.no_grad():
            val_preds = model(t_X_val_sbert, t_X_val_tab)
            val_loss = criterion(val_preds, t_y_val).item()

            # Move to CPU for metric calculation
            val_preds_np = val_preds.cpu().numpy()
            try:
                val_auc = roc_auc_score(y_val, val_preds_np)
            except ValueError:
                val_auc = 0.5

        print(
            f"Epoch {epoch+1}/{TRAIN_PARAMS['epochs']} | Train Loss: {epoch_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= TRAIN_PARAMS["patience"]:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_val_auc}"
            )
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_mlp(model, X_tab, X_sbert):
    """
    Generates probabilities using the trained MLP.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    t_X_tab = torch.tensor(X_tab, dtype=torch.float32).to(device)
    t_X_sbert = torch.tensor(X_sbert, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(t_X_sbert, t_X_tab)

    return preds.cpu().numpy().flatten()


def run_training(data):
    """
    Main orchestration function.

    Args:
        data (dict): Dictionary containing processed datasets.
    """
    set_seed()

    print("=" * 40)
    print("Starting Hybrid Ensemble Training")
    print("=" * 40)

    # ---------------------------------------------------------
    # 1. Train Learner A: Augmented Random Forest
    # ---------------------------------------------------------
    print("\n[Learner A] Training Random Forest...")
    rf_model = train_rf(data["rf_train_tab"], data["rf_train_tfidf"], data["y_train"])

    # Evaluate RF on Validation (Optional, for logging)
    rf_val_probs = predict_rf(rf_model, data["rf_val_tab"], data["rf_val_tfidf"])
    rf_val_auc = roc_auc_score(data["y_val"], rf_val_probs)
    print(f"[Learner A] Validation AUC: {rf_val_auc}")

    # ---------------------------------------------------------
    # 2. Train Learner B: Dual-Branch MLP
    # ---------------------------------------------------------
    print("\n[Learner B] Training Dual-Branch MLP...")
    mlp_model = train_mlp(
        data["mlp_train_tab"],
        data["mlp_train_sbert"],
        data["y_train"],
        data["mlp_val_tab"],
        data["mlp_val_sbert"],
        data["y_val"],
    )

    # ---------------------------------------------------------
    # 3. Inference and Ensemble
    # ---------------------------------------------------------
    print("\nGenerating Test Predictions...")

    # RF Predictions
    rf_test_probs = predict_rf(rf_model, data["rf_test_tab"], data["rf_test_tfidf"])

    # MLP Predictions
    mlp_test_probs = predict_mlp(
        mlp_model, data["mlp_test_tab"], data["mlp_test_sbert"]
    )

    # Simple Weighted Average (0.5 / 0.5)
    final_probs = 0.5 * rf_test_probs + 0.5 * mlp_test_probs

    print(f"RF Mean Prob: {rf_test_probs.mean():.4f}")
    print(f"MLP Mean Prob: {mlp_test_probs.mean():.4f}")
    print(f"Ensemble Mean Prob: {final_probs.mean():.4f}")

    # ---------------------------------------------------------
    # 4. Save Submission
    # ---------------------------------------------------------
    save_submission(data["test_ids"], final_probs)
    print("Training and inference complete.")
