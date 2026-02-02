import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed

# =========================================================================
# Stream A: Topic-Augmented Random Forest
# =========================================================================


class TopicAugmentedRF:
    """
    Wrapper for the Topic-Augmented Random Forest.
    This model utilizes a combination of:
    1. Lexical Features (TF-IDF)
    2. Metadata Features (Numerical + Ratios)
    3. Discrete Semantic Topics (K-Means cluster ratios)
    4. Semantic Consistency Scores

    The actual feature assembly is handled upstream. This class manages
    configuration and standard sklearn interface.
    """

    def __init__(self):
        self.params = Config.RF_PARAMS
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X, y):
        """
        Fits the Random Forest model.
        Args:
            X (np.ndarray or sparse matrix): Concatenated feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.
        Args:
            X (np.ndarray or sparse matrix): Feature matrix.
        Returns:
            np.ndarray: Probabilities for the positive class (column 1).
        """
        # Return probability of class 1 (Pizza Received)
        return self.model.predict_proba(X)[:, 1]


# =========================================================================
# Stream B: Attention-Gated MLP Components
# =========================================================================


class HistoryAttention(nn.Module):
    """
    Dot-Product Attention Mechanism.
    Computes context vector from history embeddings based on relevance to the request.
    Query: Request Embedding
    Keys/Values: History Embeddings
    """

    def __init__(self, embedding_dim):
        super(HistoryAttention, self).__init__()
        self.scale = np.sqrt(embedding_dim)

    def forward(self, request_emb, history_emb, mask):
        """
        Args:
            request_emb: (Batch, Dim)
            history_emb: (Batch, Seq_Len, Dim)
            mask: (Batch, Seq_Len) - 1 for valid, 0 for padding
        Returns:
            context: (Batch, Dim)
        """
        # Expand query to (Batch, 1, Dim)
        query = request_emb.unsqueeze(1)

        # Compute scores: (B, 1, D) @ (B, D, S) -> (B, 1, S)
        # Transpose history to (Batch, Dim, Seq_Len)
        keys = history_emb.transpose(1, 2)

        scores = torch.bmm(query, keys) / self.scale
        scores = scores.squeeze(1)  # (Batch, Seq_Len)

        # Apply mask (set padded positions to -inf)
        # mask is 1 for valid, 0 for pad.
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=1)  # (Batch, Seq_Len)

        # Compute Context: (B, 1, S) @ (B, S, D) -> (B, 1, D)
        weights_expanded = attn_weights.unsqueeze(1)
        context = torch.bmm(weights_expanded, history_emb)
        context = context.squeeze(1)  # (Batch, Dim)

        return context


class AttentionGatedMLP(nn.Module):
    """
    Neural Network with:
    1. Request Branch (SBERT)
    2. History Branch (Attention over Subreddits)
    3. Metadata Branch (Dense Features)
    4. Gated Fusion (Metadata gates Semantic features)
    """

    def __init__(self, metadata_dim):
        super(AttentionGatedMLP, self).__init__()

        params = Config.MLP_PARAMS
        emb_dim = params["embedding_dim"]
        hidden_dim = params["hidden_dim"]

        # 1. Semantic Branches
        self.embedding_dropout = nn.Dropout(params["dropout"])
        self.attention = HistoryAttention(emb_dim)

        # 2. Metadata Branch
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(params["meta_dropout"]),
        )

        # 3. Gating Mechanism
        # Gate controls the concatenated semantic vector (Request + Context)
        # Semantic Vector Dim = emb_dim * 2
        self.semantic_dim = emb_dim * 2
        self.gate_layer = nn.Linear(hidden_dim, self.semantic_dim)

        # 4. Final Classifier
        # Input: Gated Semantic (sem_dim) + Metadata Features (hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, metadata, request_emb, history_emb, history_mask):
        # --- Branch 1 & 2: Semantics ---
        # Apply dropout to embeddings
        req_emb = self.embedding_dropout(request_emb)
        hist_emb = self.embedding_dropout(history_emb)

        # Attention
        context = self.attention(req_emb, hist_emb, history_mask)

        # Concatenate Request + Context
        semantic_feat = torch.cat([req_emb, context], dim=1)  # (B, 768)

        # --- Branch 3: Metadata ---
        meta_feat = self.meta_mlp(metadata)  # (B, 256)

        # --- Gated Fusion ---
        # Generate Gate from Metadata
        gate = torch.sigmoid(self.gate_layer(meta_feat))  # (B, 768)

        # Apply Gate
        gated_semantic = semantic_feat * gate

        # --- Classification ---
        # Combine Gated Semantics with Metadata Features
        combined = torch.cat([gated_semantic, meta_feat], dim=1)
        logits = self.classifier(combined)

        return logits


# =========================================================================
# Training Utility
# =========================================================================


def train_mlp_model(model, train_loader, val_loader):
    """
    Trains the AttentionGatedMLP model with Early Stopping.

    Args:
        model: The PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.

    Returns:
        model: The best model state (loaded).
        history: Dictionary of training history.
    """
    set_seed(Config.SEED)

    device = Config.DEVICE
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.MLP_PARAMS["lr"],
        weight_decay=Config.MLP_PARAMS["weight_decay"],
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    patience_limit = Config.MLP_PARAMS["patience"]

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    print(f"Starting MLP training on {device}...")

    for epoch in range(Config.MLP_PARAMS["epochs"]):
        # --- Training ---
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Move data to device
            meta = batch["metadata"].to(device)
            req = batch["request_emb"].to(device)
            hist = batch["history_emb"].to(device)
            mask = batch["history_mask"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()

            logits = model(meta, req, hist, mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * meta.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                meta = batch["metadata"].to(device)
                req = batch["request_emb"].to(device)
                hist = batch["history_emb"].to(device)
                mask = batch["history_mask"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                logits = model(meta, req, hist, mask)
                loss = criterion(logits, labels)

                val_running_loss += loss.item() * meta.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                targets = labels.cpu().numpy()

                all_preds.extend(probs)
                all_targets.extend(targets)

        val_loss = val_running_loss / len(val_loader.dataset)
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5

        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch+1}/{Config.MLP_PARAMS['epochs']} - "
            f"Train Loss: {epoch_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
