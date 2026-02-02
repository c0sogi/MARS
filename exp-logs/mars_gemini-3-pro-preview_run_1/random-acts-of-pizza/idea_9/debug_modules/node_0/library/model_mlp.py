import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
import os

from library.config import Config
from library.utils import set_seed

# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Attention-Gated MLP.
    """

    def __init__(self, text_embs, sub_embs, metadata, labels=None):
        """
        Args:
            text_embs (np.ndarray): Request text embeddings (N, 384).
            sub_embs (np.ndarray): Subreddit history embeddings (N, Seq_Len, 384).
            metadata (np.ndarray): Numerical metadata features (N, Num_Meta).
            labels (np.ndarray, optional): Target labels (N,).
        """
        self.text_embs = torch.tensor(text_embs, dtype=torch.float32)
        self.sub_embs = torch.tensor(sub_embs, dtype=torch.float32)
        self.metadata = torch.tensor(metadata, dtype=torch.float32)

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.text_embs)

    def __getitem__(self, idx):
        item = {
            "text_emb": self.text_embs[idx],
            "sub_emb": self.sub_embs[idx],
            "metadata": self.metadata[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class SubredditAttention(nn.Module):
    """
    Parameter-Free Dot-Product Attention Mechanism.
    Computes importance of historical subreddits based on current request semantics.
    """

    def __init__(self):
        super(SubredditAttention, self).__init__()

    def forward(self, query, keys, mask=None):
        """
        Args:
            query: Request embedding (Batch, Dim)
            keys: Subreddit history embeddings (Batch, Seq_Len, Dim)
            mask: Boolean mask indicating valid history entries (Batch, Seq_Len)

        Returns:
            context: Weighted sum of keys (Batch, Dim)
            weights: Attention weights (Batch, Seq_Len)
        """
        # Dot product attention: (Batch, Seq_Len)
        # query.unsqueeze(1) -> (Batch, 1, Dim)
        # keys -> (Batch, Seq_Len, Dim)
        # Element-wise mult -> sum over dim -> dot product
        scores = (query.unsqueeze(1) * keys).sum(dim=-1)

        if mask is not None:
            # Mask padding with very large negative number
            scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        # weights.unsqueeze(-1) -> (Batch, Seq_Len, 1)
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)

        return context, weights


class GatedFusionMLP(nn.Module):
    """
    Hybrid Neural Network with:
    1. Attention-weighted Semantic Branch
    2. Metadata Branch
    3. Credibility Gating Mechanism
    """

    def __init__(self, meta_input_dim, params):
        super(GatedFusionMLP, self).__init__()

        emb_dim = Config.SBERT_EMBEDDING_DIM
        hidden_dim = params["hidden_dim"]
        dropout_rate = params["dropout_rate"]
        meta_dropout = params["metadata_dropout_rate"]

        # --- Branch 1: Semantics (Request + Attention Context) ---
        self.attention = SubredditAttention()
        self.sem_dropout = nn.Dropout(dropout_rate)

        # Combined semantic dimension (Request + Context)
        self.sem_dim = emb_dim * 2

        # --- Branch 2: Metadata ---
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(meta_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # --- Gating Mechanism ---
        # "Credibility Gate": Uses metadata to gate the semantic information
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.sem_dim), nn.Sigmoid()
        )

        # --- Final Classification Head ---
        # Input: Gated Semantics + Encoded Metadata
        final_input_dim = self.sem_dim + hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_emb, sub_emb, metadata):
        # 1. Semantic Processing
        # Create mask for subreddit history (assuming zero-vectors are padding)
        # Check if sum of absolute values in embedding dim is > 0
        sub_mask = sub_emb.abs().sum(dim=-1) > 1e-6

        context_emb, _ = self.attention(text_emb, sub_emb, sub_mask)

        # Concatenate Request + Context
        sem_feat = torch.cat([text_emb, context_emb], dim=1)
        sem_feat = self.sem_dropout(sem_feat)

        # 2. Metadata Processing
        meta_feat = self.meta_encoder(metadata)

        # 3. Gated Fusion
        gate = self.gate_proj(meta_feat)
        gated_sem_feat = sem_feat * gate

        # 4. Classification
        combined = torch.cat([gated_sem_feat, meta_feat], dim=1)
        logits = self.classifier(combined)

        return torch.sigmoid(logits).squeeze(-1)


# ------------------------------------------------------------------------------
# Training Function
# ------------------------------------------------------------------------------


def train_mlp(
    train_text, train_subs, train_meta, train_y, val_text, val_subs, val_meta, val_y
):
    """
    Trains the Attention-Gated MLP with early stopping.
    """
    set_seed()
    device = Config.DEVICE
    print(f"Stream B: Training on device {device}")

    # Prepare Datasets & Loaders
    train_dataset = PizzaDataset(train_text, train_subs, train_meta, train_y)
    val_dataset = PizzaDataset(val_text, val_subs, val_meta, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # Avoid BatchNorm error on single sample
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize Model
    meta_dim = train_meta.shape[1]
    model = GatedFusionMLP(meta_dim, Config.MLP_PARAMS).to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCELoss()

    # Early Stopping Tracking
    best_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Stream B: Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            txt = batch["text_emb"].to(device)
            sub = batch["sub_emb"].to(device)
            meta = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(txt, sub, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * txt.size(0)

        epoch_loss = train_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                txt = batch["text_emb"].to(device)
                sub = batch["sub_emb"].to(device)
                meta = batch["metadata"].to(device)
                labels = batch["label"].to(device)

                outputs = model(txt, sub, meta)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        # --- Logging & Early Stopping ---
        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Loss: {epoch_loss:.6f} | Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Stream B: Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_wts)
    print(f"Stream B: Best Validation AUC: {best_auc}")

    return model


# ------------------------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------------------------


def predict_mlp(model, test_text, test_subs, test_meta):
    """
    Generates predictions using the trained MLP.
    """
    set_seed()
    device = Config.DEVICE
    model.eval()
    model.to(device)

    dataset = PizzaDataset(test_text, test_subs, test_meta)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    all_preds = []

    with torch.no_grad():
        for batch in loader:
            txt = batch["text_emb"].to(device)
            sub = batch["sub_emb"].to(device)
            meta = batch["metadata"].to(device)

            outputs = model(txt, sub, meta)
            all_preds.extend(outputs.cpu().numpy())

    return np.array(all_preds)
