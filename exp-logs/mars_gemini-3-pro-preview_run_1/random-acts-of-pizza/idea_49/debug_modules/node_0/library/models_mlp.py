import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import os

from library.config import Config


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request data.
    Handles semantic features (SBERT embeddings) and numerical metadata.
    """

    def __init__(self, features, labels=None):
        self.title = torch.FloatTensor(features["title"])
        self.body = torch.FloatTensor(features["body"])
        self.history = torch.FloatTensor(features["history"])
        self.history_mask = torch.FloatTensor(features["history_mask"])
        self.centroid = torch.FloatTensor(features["centroid"])
        self.metadata = torch.FloatTensor(features["metadata"])

        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        item = {
            "title": self.title[idx],
            "body": self.body[idx],
            "history": self.history[idx],
            "history_mask": self.history_mask[idx],
            "centroid": self.centroid[idx],
            "metadata": self.metadata[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class DualQueryAttention(nn.Module):
    """
    Attention mechanism with two heads:
    1. Query = Title, Key/Value = User History
    2. Query = Body, Key/Value = User History
    """

    def __init__(self, embed_dim):
        super(DualQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        # Projections for stability
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

    def forward(self, query, history, mask):
        """
        Args:
            query: (B, D)
            history: (B, Seq, D)
            mask: (B, Seq) - 1 for valid, 0 for pad
        Returns:
            context: (B, D)
        """
        # Expand query to (B, 1, D)
        Q = self.query_proj(query).unsqueeze(1)
        K = self.key_proj(history)  # (B, Seq, D)
        V = self.value_proj(history)  # (B, Seq, D)

        # Scores: (B, 1, D) @ (B, D, Seq) -> (B, 1, Seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask: set pad positions to -inf
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        # Context: (B, 1, Seq) @ (B, Seq, D) -> (B, 1, D)
        context = torch.matmul(attn_weights, V)

        return context.squeeze(1)  # (B, D)


class OrthogonalFiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Modulates the semantic feature vector 'x' using affine parameters derived
    strictly from metadata 'z'.
    """

    def __init__(self, input_dim, cond_dim):
        super(OrthogonalFiLM, self).__init__()
        self.gamma = nn.Linear(cond_dim, input_dim)
        self.beta = nn.Linear(cond_dim, input_dim)

        # Initialize gamma to 0 (so 1+gamma = 1) and beta to 0 for identity start
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, z):
        # x: (B, input_dim) - Semantic features
        # z: (B, cond_dim) - Metadata

        gamma = self.gamma(z)
        beta = self.beta(z)

        return (1 + gamma) * x + beta


class HybridPizzaNetwork(nn.Module):
    """
    Main Neural Network Architecture.
    Combines SBERT embeddings, Dual-Query Attention over history, and Global Centroids.
    Fuses these with metadata using Orthogonal FiLM.
    """

    def __init__(self, metadata_dim):
        super(HybridPizzaNetwork, self).__init__()

        emb_dim = Config.MLP_EMBEDDING_DIM
        hidden_dim = Config.MLP_HIDDEN_DIM
        dropout_emb = Config.MLP_DROPOUT_EMB
        dropout_dense = Config.MLP_DROPOUT_DENSE

        # Regularization
        self.dropout_layer = nn.Dropout(dropout_emb)

        # Attention Modules
        self.attn_title_hist = DualQueryAttention(emb_dim)
        self.attn_body_hist = DualQueryAttention(emb_dim)

        # Semantic Feature Vector Construction
        # Components: Title(384) + Body(384) + Hist_Ctx_Title(384) + Hist_Ctx_Body(384) + Centroid(384)
        # Total Semantic Dim = 384 * 5 = 1920
        self.semantic_dim = emb_dim * 5

        # FiLM Layer
        self.film = OrthogonalFiLM(self.semantic_dim, metadata_dim)

        # Fusion Classifier
        # Input: Modulated Semantics (1920) + Metadata (metadata_dim) (Skip Connection)
        fusion_dim = self.semantic_dim + metadata_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, title, body, history, history_mask, centroid, metadata):
        # 1. Apply Dropout to raw embeddings
        title = self.dropout_layer(title)
        body = self.dropout_layer(body)
        history = self.dropout_layer(history)
        centroid = self.dropout_layer(centroid)

        # 2. Dual Attention
        ctx_title = self.attn_title_hist(title, history, history_mask)
        ctx_body = self.attn_body_hist(body, history, history_mask)

        # 3. Construct Semantic Vector x
        x = torch.cat([title, body, ctx_title, ctx_body, centroid], dim=1)

        # 4. Apply FiLM modulation using metadata z
        x_mod = self.film(x, metadata)

        # 5. Skip Connection / Fusion
        # Concatenate modulated semantics with original metadata
        combined = torch.cat([x_mod, metadata], dim=1)

        # 6. Classifier
        logits = self.classifier(combined)
        return logits


def train_model(
    train_features, train_labels, val_features, val_labels, input_metadata_dim
):
    """
    Trains the HybridPizzaNetwork with Early Stopping.
    """
    set_seed(Config.RANDOM_SEED)
    device = torch.device(Config.DEVICE)

    # Prepare Datasets
    train_dataset = PizzaDataset(train_features, train_labels)
    val_dataset = PizzaDataset(val_features, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False
    )

    # Initialize Model
    model = HybridPizzaNetwork(metadata_dim=input_metadata_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.MLP_LEARNING_RATE,
        weight_decay=Config.MLP_WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting MLP training on {device}...")

    for epoch in range(Config.MLP_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Move to device
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            history = batch["history"].to(device)
            mask = batch["history_mask"].to(device)
            centroid = batch["centroid"].to(device)
            meta = batch["metadata"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(title, body, history, mask, centroid, meta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * title.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                title = batch["title"].to(device)
                body = batch["body"].to(device)
                history = batch["history"].to(device)
                mask = batch["history_mask"].to(device)
                centroid = batch["centroid"].to(device)
                meta = batch["metadata"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                logits = model(title, body, history, mask, centroid, meta)
                loss = criterion(logits, labels)
                val_loss += loss.item() * title.size(0)

                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.MLP_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict(model, test_features):
    """
    Generates predictions for the test set.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    test_dataset = PizzaDataset(test_features)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False
    )

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            history = batch["history"].to(device)
            mask = batch["history_mask"].to(device)
            centroid = batch["centroid"].to(device)
            meta = batch["metadata"].to(device)

            logits = model(title, body, history, mask, centroid, meta)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().flatten())

    return np.array(all_preds)
