import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, ensure_dir, save_model
from library.dataset import PizzaDataset


class AttentionHead(nn.Module):
    """
    Simple Dot-Product Attention Head.
    Computes attention between a Query vector (B, D) and a Key sequence (B, S, D).
    """

    def __init__(self, input_dim):
        super().__init__()
        self.scale = input_dim**-0.5

    def forward(self, query, key, value, mask=None):
        # query: (B, D)
        # key: (B, S, D)
        # value: (B, S, D)
        # mask: (B, S) - 1 for valid, 0 for pad

        # Expand query to (B, 1, D)
        q = query.unsqueeze(1)

        # Transpose key to (B, D, S)
        k_t = key.transpose(1, 2)

        # Compute scores: (B, 1, D) @ (B, D, S) -> (B, 1, S)
        scores = torch.bmm(q, k_t) * self.scale

        if mask is not None:
            # Expand mask to (B, 1, S)
            mask_u = mask.unsqueeze(1)
            # Apply additive masking
            scores = scores.masked_fill(mask_u == 0, -1e9)

        # Softmax over sequence dimension
        weights = torch.softmax(scores, dim=-1)  # (B, 1, S)

        # Weighted sum: (B, 1, S) @ (B, S, D) -> (B, 1, D)
        context = torch.bmm(weights, value)

        # Squeeze back to (B, D)
        return context.squeeze(1)


class DualAttentionCentroidMLP(nn.Module):
    """
    Stream B: Centroid-Augmented Dual-Query MLP.
    Combines selective attention (Title/Body -> History) with global persona injection (Centroid).
    Uses a Credibility Gate derived from metadata.
    """

    def __init__(self, meta_dim=7, align_dim=2):
        super().__init__()

        self.emb_dim = Config.EMBEDDING_DIM

        # 1. Attention Branches
        self.att_topic = AttentionHead(self.emb_dim)  # Query: Title
        self.att_narrative = AttentionHead(self.emb_dim)  # Query: Body

        # 2. Fusion Dimension Calculation
        # Components: Title(384) + Body(384) + TopicCtx(384) + NarrativeCtx(384) + Centroid(384) + Alignment(2)
        self.fusion_dim = (self.emb_dim * 5) + align_dim

        # 3. Credibility Gate (Branch 5)
        # Projects Metadata -> Fusion Dimension to gate the signal
        self.gate_net = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.fusion_dim),
            nn.Sigmoid(),
        )

        # 4. Main Classifier
        layers = []
        input_dim = self.fusion_dim

        for h_dim in Config.MLP_HIDDEN_DIMS:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.MLP_DROPOUT_DENSE))
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, 1))
        self.classifier = nn.Sequential(*layers)

        # Regularization (Dropout on Embeddings)
        self.dropout_emb = nn.Dropout(Config.MLP_DROPOUT_EMB)

    def forward(self, title, body, hist_seq, hist_mask, centroid, meta, align):
        # Apply dropout to raw embeddings
        title = self.dropout_emb(title)
        body = self.dropout_emb(body)
        hist_seq = self.dropout_emb(hist_seq)
        centroid = self.dropout_emb(centroid)

        # Branch 3: Dual-Query Attention
        # Head A: Topic Context (Title -> History)
        ctx_topic = self.att_topic(
            query=title, key=hist_seq, value=hist_seq, mask=hist_mask
        )

        # Head B: Narrative Context (Body -> History)
        ctx_narrative = self.att_narrative(
            query=body, key=hist_seq, value=hist_seq, mask=hist_mask
        )

        # Fusion: Concatenate all semantic and structural signals
        # (B, 384*5 + 2)
        fusion_vec = torch.cat(
            [title, body, ctx_topic, ctx_narrative, centroid, align], dim=1
        )

        # Gating: Modulate fusion vector based on metadata credibility
        gate = self.gate_net(meta)
        gated_vec = fusion_vec * gate

        # Classification
        logits = self.classifier(gated_vec)
        return logits


def train_mlp_model(features_dict, save_path=None):
    """
    Trains the MLP model.

    Args:
        features_dict (dict): Dictionary containing all features (train/val/test).
        save_path (str): Path to save the best model.

    Returns:
        model: Trained PyTorch model.
    """
    set_seed(Config.RANDOM_STATE)
    device = Config.DEVICE

    # Prepare Datasets
    print("Preparing MLP Datasets...")
    train_dataset = PizzaDataset(features_dict, split="train")
    val_dataset = PizzaDataset(features_dict, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.MLP_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    # Initialize Model
    # Metadata dim is 7, Alignment dim is 2 (based on features.py logic)
    model = DualAttentionCentroidMLP(meta_dim=7, align_dim=2)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.MLP_LEARNING_RATE,
        weight_decay=Config.MLP_WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting MLP Training on {device}...")
    print(f"  Epochs: {Config.MLP_EPOCHS}")
    print(f"  Patience: {Config.MLP_PATIENCE}")

    for epoch in range(Config.MLP_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Move batch to device
            title = batch["title_emb"].to(device)
            body = batch["body_emb"].to(device)
            hist_seq = batch["history_seqs"].to(device)
            hist_mask = batch["history_mask"].to(device)
            centroid = batch["centroid"].to(device)
            meta = batch["metadata"].to(device)
            align = batch["alignment"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()

            logits = model(title, body, hist_seq, hist_mask, centroid, meta, align)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * title.size(0)

        avg_train_loss = train_loss / len(train_dataset)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                title = batch["title_emb"].to(device)
                body = batch["body_emb"].to(device)
                hist_seq = batch["history_seqs"].to(device)
                hist_mask = batch["history_mask"].to(device)
                centroid = batch["centroid"].to(device)
                meta = batch["metadata"].to(device)
                align = batch["alignment"].to(device)
                labels = batch["label"].to(device)

                logits = model(title, body, hist_seq, hist_mask, centroid, meta, align)
                probs = torch.sigmoid(logits).squeeze(1)

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        # Print metrics
        print(
            f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc}"
        )

        # --- Early Stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= Config.MLP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # Save and Load Best Model
    if best_model_state is not None:
        if save_path:
            ensure_dir(save_path)
            print(f"Saving best MLP model to {save_path}...")
            torch.save(best_model_state, save_path)

        model.load_state_dict(best_model_state)

    return model


def predict_mlp(model, features_dict, split="test"):
    """
    Generates predictions using the trained MLP model.

    Args:
        model: Trained DualAttentionCentroidMLP.
        features_dict (dict): Feature dictionary.
        split (str): 'train', 'val', or 'test'.

    Returns:
        np.ndarray: Probability predictions.
    """
    device = Config.DEVICE
    model.eval()
    model.to(device)

    dataset = PizzaDataset(features_dict, split=split)
    loader = DataLoader(
        dataset,
        batch_size=Config.MLP_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    all_probs = []

    with torch.no_grad():
        for batch in loader:
            title = batch["title_emb"].to(device)
            body = batch["body_emb"].to(device)
            hist_seq = batch["history_seqs"].to(device)
            hist_mask = batch["history_mask"].to(device)
            centroid = batch["centroid"].to(device)
            meta = batch["metadata"].to(device)
            align = batch["alignment"].to(device)

            logits = model(title, body, hist_seq, hist_mask, centroid, meta, align)
            probs = torch.sigmoid(logits).squeeze(1)

            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)
