import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import (
    RF_N_ESTIMATORS,
    RF_MIN_SAMPLES_LEAF,
    RF_CLASS_WEIGHT,
    RF_N_JOBS,
    MLP_HIDDEN_DIM,
    MLP_DROPOUT,
    MLP_DROPOUT_DENSE,
    MLP_LEARNING_RATE,
    MLP_WEIGHT_DECAY,
    MLP_NUM_EPOCHS,
    MLP_PATIENCE,
    DEVICE,
    CACHE_DIR,
)


class InteractionRandomForest:
    """
    Stream A: Interaction-Enhanced Top-K Random Forest.
    Wraps sklearn's RandomForestClassifier with specific configurations for
    handling sparse interaction features and class imbalance.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight=RF_CLASS_WEIGHT,
            n_jobs=RF_N_JOBS,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DualHistoryAttention(nn.Module):
    """
    Computes attention over user history using a specific query (Title or Body).
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.scale = embed_dim**-0.5

    def forward(self, query, history, mask):
        """
        Args:
            query: (B, D) - The vector to query with (e.g., Title embedding)
            history: (B, L, D) - Sequence of history embeddings
            mask: (B, L) - 1 for valid history, 0 for padding
        Returns:
            context: (B, D) - Weighted sum of history
        """
        # Expand query to (B, 1, D) for batch matrix multiplication
        q = query.unsqueeze(1)

        # Calculate attention scores: (B, 1, D) @ (B, D, L) -> (B, 1, L)
        scores = torch.bmm(q, history.transpose(1, 2)) * self.scale

        # Apply mask (set padding positions to -inf)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, L)
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)

        # Context: (B, 1, L) @ (B, L, D) -> (B, 1, D)
        context = torch.bmm(weights, history)

        return context.squeeze(1)


class FiLMLayer(nn.Module):
    """
    Orthogonal Feature-wise Linear Modulation (FiLM).
    Modulates the semantic feature stream 'x' based on metadata 'z'.
    Output = (1 + gamma(z)) * x + beta(z)
    """

    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.gamma_net = nn.Linear(condition_dim, feature_dim)
        self.beta_net = nn.Linear(condition_dim, feature_dim)

        # Initialize to identity mapping
        # gamma -> 0 (scale -> 1), beta -> 0 (shift -> 0)
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, x, z):
        gamma = self.gamma_net(z)
        beta = self.beta_net(z)
        return (1 + gamma) * x + beta


class PizzaFiLMMLP(nn.Module):
    """
    Stream B: Orthogonal FiLM-Conditioned Dual-Query MLP.
    Fuses semantic content, user history, and metadata using FiLM.
    """

    def __init__(self, metadata_dim, embed_dim=384, hidden_dim=MLP_HIDDEN_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        # Attention Modules
        self.title_attention = DualHistoryAttention(embed_dim)
        self.body_attention = DualHistoryAttention(embed_dim)

        # Fusion Dimension Calculation:
        # Title(D) + Body(D) + ContextTitle(D) + ContextBody(D) + Centroid(D) + AlignScalars(2)
        self.fusion_dim = (embed_dim * 5) + 2

        # FiLM Layer for Metadata Injection
        self.film = FiLMLayer(self.fusion_dim, metadata_dim)

        # Classification Head
        self.mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout_emb = nn.Dropout(MLP_DROPOUT)

    def forward(self, title, body, history, history_mask, centroid, metadata):
        # Apply dropout to embeddings
        title = self.dropout_emb(title)
        body = self.dropout_emb(body)
        history = self.dropout_emb(history)
        centroid = self.dropout_emb(centroid)

        # 1. Dual-Query Attention
        context_title = self.title_attention(title, history, history_mask)
        context_body = self.body_attention(body, history, history_mask)

        # 2. Compute Alignment Scalars (Consistency)
        # Dot product between content and centroid: (B, D) * (B, D) -> sum -> (B, 1)
        align_title = (title * centroid).sum(dim=1, keepdim=True)
        align_body = (body * centroid).sum(dim=1, keepdim=True)

        # 3. Concatenate all semantic signals
        x = torch.cat(
            [
                title,
                body,
                context_title,
                context_body,
                centroid,
                align_title,
                align_body,
            ],
            dim=1,
        )

        # 4. Apply Orthogonal FiLM Conditioning
        # Modulate high-dim semantic features 'x' with low-dim metadata 'metadata'
        x_modulated = self.film(x, metadata)

        # 5. MLP Head
        logits = self.mlp(x_modulated)

        return logits


def train_mlp_model(train_loader, val_loader, metadata_dim):
    """
    Training loop for the PizzaFiLMMLP model with Early Stopping.
    """
    model = PizzaFiLMMLP(metadata_dim=metadata_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting MLP Training on {DEVICE}...")

    for epoch in range(MLP_NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            # Move data to device
            title = batch["title_emb"].to(DEVICE)
            body = batch["body_emb"].to(DEVICE)
            hist = batch["history_emb"].to(DEVICE)
            mask = batch["history_mask"].to(DEVICE)
            cent = batch["centroid_emb"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(title, body, hist, mask, cent, meta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * title.size(0)
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
        except ValueError:
            train_auc = 0.5

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                title = batch["title_emb"].to(DEVICE)
                body = batch["body_emb"].to(DEVICE)
                hist = batch["history_emb"].to(DEVICE)
                mask = batch["history_mask"].to(DEVICE)
                cent = batch["centroid_emb"].to(DEVICE)
                meta = batch["metadata"].to(DEVICE)
                labels = batch["label"].to(DEVICE).unsqueeze(1)

                logits = model(title, body, hist, mask, cent, meta)
                loss = criterion(logits, labels)

                val_loss += loss.item() * title.size(0)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5

        print(
            f"Epoch {epoch+1}/{MLP_NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.10f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.10f}"
        )

        # --- Early Stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model to cache
            torch.save(best_model_state, os.path.join(CACHE_DIR, "best_mlp.pth"))
        else:
            patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
