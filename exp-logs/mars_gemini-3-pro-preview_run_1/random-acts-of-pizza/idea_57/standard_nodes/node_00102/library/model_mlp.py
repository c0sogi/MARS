import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from library import config, utils

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------
# Feature Dimensions based on feature_engineering.py logic
# Semantic: 3 * 384 (SBERT) + 2 (Consistency) = 1154
SEMANTIC_DIM = 1154
# Metadata: 9 numerical features
META_DIM = 9
# Top-K: 50 binary indicators
TOPK_DIM = 50


# -----------------------------------------------------------------------------
# Dataset Definition
# -----------------------------------------------------------------------------
class PizzaDataset(Dataset):
    """
    Custom Dataset to handle the split of the concatenated feature vector
    into Semantic, Metadata, and Top-K components.
    """

    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

        # Pre-calculate slice indices
        self.sem_end = SEMANTIC_DIM
        self.meta_end = SEMANTIC_DIM + META_DIM

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Slice the input vector
        # [0 : 1154] -> Semantic
        sem = self.X[idx, : self.sem_end]
        # [1154 : 1163] -> Metadata
        meta = self.X[idx, self.sem_end : self.meta_end]
        # [1163 : ] -> TopK
        topk = self.X[idx, self.meta_end :]

        if self.y is not None:
            return sem, meta, topk, self.y[idx]
        return sem, meta, topk


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------
class TopologyAwareMLP(nn.Module):
    """
    Topology-Aware Non-Linear Skip-Gated MLP.

    Structure:
    1. Semantic Branch: Projects high-dim text/history features to hidden space.
    2. Control Branch: Uses metadata to generate a non-linear gate.
    3. Fusion: Gated Semantic + Raw Metadata + Top-K Bias.
    """

    def __init__(self, hidden_dim=256, dropout_emb=0.5, dropout_dense=0.2):
        super(TopologyAwareMLP, self).__init__()

        # Branch 1: Semantic Content Projection
        self.semantic_proj = nn.Sequential(
            nn.Dropout(p=dropout_emb),
            nn.Linear(SEMANTIC_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_dense),
        )

        # Branch 2: Reliability Control Gate
        # Generates a gate of size `hidden_dim` from `META_DIM` inputs
        self.control_gate = nn.Sequential(
            nn.Linear(META_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),  # Gate values between 0 and 1
        )

        # Final Fusion Layer
        # Concatenates: Gated Semantic (hidden_dim) + Raw Meta (META_DIM) + TopK (TOPK_DIM)
        fusion_input_dim = hidden_dim + META_DIM + TOPK_DIM

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_dense),
            nn.Linear(hidden_dim // 2, 1),  # Logits
        )

    def forward(self, sem, meta, topk):
        # 1. Project Semantic Features
        h_sem = self.semantic_proj(sem)

        # 2. Generate Control Gate
        gate = self.control_gate(meta)

        # 3. Apply Gate (Modulation)
        # Element-wise multiplication: Reliability modulates Content
        h_gated = h_sem * gate

        # 4. Topology-Aware Fusion
        # Concatenate Gated Content, Raw Control Signals, and Community Bias
        combined = torch.cat([h_gated, meta, topk], dim=1)

        # 5. Classification
        logits = self.classifier(combined)
        return logits


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_mlp_model(X_train, y_train, X_val, y_val, params=None, save_path=None):
    """
    Trains the Topology-Aware MLP model.
    """
    # Set seed for reproducibility
    utils.set_seed(config.RANDOM_STATE)

    # Default Parameters
    if params is None:
        params = {}

    hidden_dim = params.get("hidden_dim", config.MLP_HIDDEN_DIM)
    dropout_emb = params.get("dropout_emb", config.MLP_DROPOUT_EMB)
    dropout_dense = params.get("dropout_dense", config.MLP_DROPOUT_DENSE)
    lr = params.get("learning_rate", config.MLP_LEARNING_RATE)
    weight_decay = params.get("weight_decay", config.MLP_WEIGHT_DECAY)
    epochs = params.get("epochs", config.MLP_EPOCHS)
    patience = params.get("patience", config.MLP_PATIENCE)
    batch_size = params.get("batch_size", config.MLP_BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP on device: {device}")

    # Prepare DataLoaders
    train_dataset = PizzaDataset(X_train, y_train)
    val_dataset = PizzaDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = TopologyAwareMLP(
        hidden_dim=hidden_dim, dropout_emb=dropout_emb, dropout_dense=dropout_dense
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for sem, meta, topk, targets in train_loader:
            sem, meta, topk, targets = (
                sem.to(device),
                meta.to(device),
                topk.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()
            logits = model(sem, meta, topk)
            loss = criterion(logits.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sem.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for sem, meta, topk, targets in val_loader:
                sem, meta, topk, targets = (
                    sem.to(device),
                    meta.to(device),
                    topk.to(device),
                    targets.to(device),
                )
                logits = model(sem, meta, topk)
                loss = criterion(logits.squeeze(), targets)
                val_loss += loss.item() * sem.size(0)

                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                # Handle single-element batch edge case where squeeze returns scalar
                if probs.ndim == 0:
                    probs = np.array([probs])

                val_preds.extend(probs)
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save intermediate best
            if save_path:
                torch.save(best_model_state, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_val_auc}"
            )
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val AUC: {best_val_auc}")

    return model, best_val_auc


# -----------------------------------------------------------------------------
# Inference Function
# -----------------------------------------------------------------------------
def predict_mlp_model(model, X_test):
    """
    Generates predictions using the trained MLP model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Use config batch size for inference or default
    batch_size = config.MLP_BATCH_SIZE
    test_dataset = PizzaDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_probs = []

    print(f"Generating predictions for {len(X_test)} samples...")

    with torch.no_grad():
        for sem, meta, topk in test_loader:
            sem, meta, topk = sem.to(device), meta.to(device), topk.to(device)
            logits = model(sem, meta, topk)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            if probs.ndim == 0:
                probs = np.array([probs])

            all_probs.extend(probs)

    return np.array(all_probs)
