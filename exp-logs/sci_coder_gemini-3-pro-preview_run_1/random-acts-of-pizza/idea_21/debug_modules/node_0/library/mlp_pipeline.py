import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
import os
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.RANDOM_SEED)


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Hybrid Ensemble MLP.
    Handles dictionary inputs containing request embeddings, history sequences, and metadata.
    """

    def __init__(self, features_dict, targets=None):
        """
        Args:
            features_dict (dict): Dictionary containing:
                - 'request_emb': np.array (N, 384)
                - 'history_seq': np.array (N, SeqLen, 384)
                - 'metadata': np.array (N, MetaDim)
            targets (np.array, optional): Binary targets (N,).
        """
        self.request_emb = torch.FloatTensor(features_dict["request_emb"])
        self.history_seq = torch.FloatTensor(features_dict["history_seq"])
        self.metadata = torch.FloatTensor(features_dict["metadata"])

        if targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.request_emb)

    def __getitem__(self, idx):
        sample = {
            "request_emb": self.request_emb[idx],
            "history_seq": self.history_seq[idx],
            "metadata": self.metadata[idx],
        }

        if self.targets is not None:
            return sample, self.targets[idx]
        return sample


class GatedAttentionNet(nn.Module):
    """
    Neural Network with Credibility-Gated Attention.

    Branches:
    1. Request Branch: Processes SBERT embedding of the request.
    2. History Branch: Uses Dot-Product Attention (Query=Request, Key=History) to extract relevant history.
    3. Metadata Branch: Processes numerical metadata to form a 'Credibility Gate'.

    Fusion:
    The semantic representation (Request + Attended History) is modulated (gated) by the
    credibility signal derived from metadata.
    """

    def __init__(
        self, meta_dim, hidden_dim=Config.MLP_HIDDEN_DIM, dropout=Config.MLP_DROPOUT
    ):
        super(GatedAttentionNet, self).__init__()

        self.sbert_dim = Config.SBERT_DIM

        # --- Branch 1: Request Semantics ---
        # Simple projection of the request embedding
        self.request_proj = nn.Linear(self.sbert_dim, hidden_dim)
        self.request_dropout = nn.Dropout(dropout)

        # --- Branch 2: History Attention ---
        # We use the raw SBERT dimension for dot-product attention to preserve semantic space
        # Query = Request (384), Key = History (384)
        # Output will be projected to hidden_dim
        self.history_proj = nn.Linear(self.sbert_dim, hidden_dim)
        self.history_dropout = nn.Dropout(dropout)

        # --- Branch 3: Metadata (Credibility) ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # --- Gating Mechanism ---
        # Generates a gate vector of size (2 * hidden_dim) to modulate the concatenated semantic vector
        self.gate_layer = nn.Linear(hidden_dim, 2 * hidden_dim)

        # --- Final Classifier ---
        # Input: Gated Semantics (2 * hidden_dim) + Metadata Features (hidden_dim)
        # We include metadata explicitly in the final layer so it can act as a direct predictor
        # independent of the semantic gating.
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, request_emb, history_seq, metadata):
        # 1. Request Feature
        # request_emb: (B, 384)
        req_feat = self.request_proj(request_emb)  # (B, H)
        req_feat = self.request_dropout(req_feat)

        # 2. History Attention
        # history_seq: (B, L, 384)
        # Query: request_emb (B, 1, 384)
        query = request_emb.unsqueeze(1)
        keys = history_seq

        # Dot Product Attention Scores: (B, 1, L)
        # Scale by sqrt(dim)
        scores = torch.bmm(query, keys.transpose(1, 2)) / (self.sbert_dim**0.5)

        # Masking: Identify padding (all zeros) in history_seq
        # Sum absolute values across embedding dim. If 0, it's padding.
        mask = history_seq.abs().sum(dim=2) > 0  # (B, L)
        mask = mask.unsqueeze(1)  # (B, 1, L)

        # Apply mask (set scores to -inf where mask is False)
        scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, L)

        # Context Vector: (B, 1, 384)
        context = torch.bmm(attn_weights, history_seq)
        context = context.squeeze(1)  # (B, 384)

        # Project context to hidden dim
        hist_feat = self.history_proj(context)  # (B, H)
        hist_feat = self.history_dropout(hist_feat)

        # 3. Metadata Feature
        meta_feat = self.meta_mlp(metadata)  # (B, H)

        # 4. Gated Fusion
        # Semantic Vector = Concat(Request, History) -> (B, 2H)
        semantic_vector = torch.cat([req_feat, hist_feat], dim=1)

        # Gate Generation -> (B, 2H)
        gate = torch.sigmoid(self.gate_layer(meta_feat))

        # Modulate Semantics
        gated_semantics = semantic_vector * gate

        # 5. Classification
        # Concat (Gated Semantics, Metadata) -> (B, 3H)
        combined = torch.cat([gated_semantics, meta_feat], dim=1)
        logits = self.classifier(combined)

        return logits.squeeze(1)


def train_mlp_model(
    train_features,
    y_train,
    val_features=None,
    y_val=None,
    batch_size=Config.MLP_BATCH_SIZE,
    epochs=Config.MLP_EPOCHS,
    lr=Config.MLP_LEARNING_RATE,
    weight_decay=Config.MLP_WEIGHT_DECAY,
    patience=Config.MLP_PATIENCE,
    device=Config.DEVICE,
):
    """
    Trains the GatedAttentionNet with early stopping.

    Args:
        train_features (dict): Dictionary of training features.
        y_train (np.array): Training labels.
        val_features (dict, optional): Validation features.
        y_val (np.array, optional): Validation labels.

    Returns:
        model: Best trained model state.
    """
    set_seed(Config.RANDOM_SEED)

    # Prepare Datasets
    train_dataset = PizzaDataset(train_features, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    if val_features is not None:
        val_dataset = PizzaDataset(val_features, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        val_loader = None

    # Initialize Model
    meta_dim = train_features["metadata"].shape[1]
    model = GatedAttentionNet(meta_dim=meta_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting MLP training on {device}...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch_data, batch_targets in train_loader:
            batch_targets = batch_targets.to(device)

            # Unpack inputs
            req = batch_data["request_emb"].to(device)
            hist = batch_data["history_seq"].to(device)
            meta = batch_data["metadata"].to(device)

            optimizer.zero_grad()
            logits = model(req, hist, meta)
            loss = criterion(logits, batch_targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_targets.size(0)

        train_loss /= len(train_dataset)

        # --- Validation ---
        val_auc = 0.0
        if val_loader:
            model.eval()
            val_preds = []
            val_targets_all = []

            with torch.no_grad():
                for batch_data, batch_targets in val_loader:
                    req = batch_data["request_emb"].to(device)
                    hist = batch_data["history_seq"].to(device)
                    meta = batch_data["metadata"].to(device)

                    logits = model(req, hist, meta)
                    probs = torch.sigmoid(logits)

                    val_preds.extend(probs.cpu().numpy())
                    val_targets_all.extend(batch_targets.numpy())

            val_auc = roc_auc_score(val_targets_all, val_preds)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f}")
            # If no validation set, save last state
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Training complete. Best Val AUC: {best_val_auc}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def predict_mlp(
    model, features_dict, batch_size=Config.MLP_BATCH_SIZE, device=Config.DEVICE
):
    """
    Generates predictions using the trained MLP model.

    Args:
        model: Trained GatedAttentionNet.
        features_dict (dict): Dictionary of features.

    Returns:
        np.array: Predicted probabilities.
    """
    model.eval()
    dataset = PizzaDataset(features_dict, targets=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []

    with torch.no_grad():
        for batch_data in loader:
            req = batch_data["request_emb"].to(device)
            hist = batch_data["history_seq"].to(device)
            meta = batch_data["metadata"].to(device)

            logits = model(req, hist, meta)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)
