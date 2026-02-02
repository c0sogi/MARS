import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from library import config, utils

# =============================================================================
# DATASET
# =============================================================================


class PizzaDataset(Dataset):
    def __init__(self, meta, title_emb, body_emb, hist_emb, hist_mask, labels=None):
        """
        Args:
            meta (np.ndarray): Scaled metadata features.
            title_emb (np.ndarray): SBERT embeddings for titles.
            body_emb (np.ndarray): SBERT embeddings for bodies.
            hist_emb (np.ndarray): Sequence embeddings for history.
            hist_mask (np.ndarray): Mask for history sequences (1=valid, 0=pad).
            labels (np.ndarray, optional): Target labels.
        """
        self.meta = torch.FloatTensor(meta)
        self.title_emb = torch.FloatTensor(title_emb)
        self.body_emb = torch.FloatTensor(body_emb)
        self.hist_emb = torch.FloatTensor(hist_emb)
        self.hist_mask = torch.FloatTensor(hist_mask)

        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = {
            "meta": self.meta[idx],
            "title": self.title_emb[idx],
            "body": self.body_emb[idx],
            "history": self.hist_emb[idx],
            "mask": self.hist_mask[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


# =============================================================================
# MODULES
# =============================================================================


class DualQueryAttention(nn.Module):
    def __init__(self, embed_dim):
        super(DualQueryAttention, self).__init__()
        self.scale = embed_dim**-0.5

    def forward(self, query, key, value, mask):
        """
        Args:
            query: (Batch, EmbedDim)
            key: (Batch, SeqLen, EmbedDim)
            value: (Batch, SeqLen, EmbedDim)
            mask: (Batch, SeqLen) - 1 for valid, 0 for pad
        Returns:
            context: (Batch, EmbedDim)
        """
        # Expand query to (Batch, 1, EmbedDim) for broadcasting
        query = query.unsqueeze(1)

        # Calculate attention scores: (Batch, 1, SeqLen)
        # Q * K^T
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale

        # Squeeze to (Batch, SeqLen)
        scores = scores.squeeze(1)

        # Apply Masking: Set masked positions to -infinity
        # mask is 1 for keep, 0 for mask. We want to fill where mask == 0.
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=1)

        # Calculate Context: (Batch, 1, SeqLen) * (Batch, SeqLen, EmbedDim) -> (Batch, 1, EmbedDim)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, value)

        # Squeeze back to (Batch, EmbedDim)
        context = context.squeeze(1)

        return context


class PizzaNet(nn.Module):
    def __init__(self, meta_dim, embed_dim=384, hidden_dim=256, dropout=0.3):
        super(PizzaNet, self).__init__()

        # 1. Feature Encoders
        self.title_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        self.body_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # 2. Dual-Query Attention Mechanism
        # We don't project history before attention to preserve raw semantics for the query
        self.attention = DualQueryAttention(embed_dim)

        # Project concatenated context (Topic Context + Narrative Context)
        self.history_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # 3. Metadata Encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # 4. Gated Fusion
        # Semantic Vector S = [Title_Enc, Body_Enc, History_Enc] -> Dim = 3 * hidden_dim
        self.semantic_dim = hidden_dim * 3

        # Gate Generator: Metadata (128) -> Gate (Semantic_Dim)
        self.gate_generator = nn.Sequential(
            nn.Linear(128, self.semantic_dim), nn.Sigmoid()
        )

        # 5. Final Classification Head
        # Input: Gated Semantic Vector + Metadata Vector
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim + 128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, meta, title, body, history, mask):
        # Encode Text
        h_title = self.title_encoder(title)
        h_body = self.body_encoder(body)

        # Dual-Query Attention
        # Head A: Topic Context (Query=Title, Key=History)
        ctx_topic = self.attention(title, history, history, mask)
        # Head B: Narrative Context (Query=Body, Key=History)
        ctx_narrative = self.attention(body, history, history, mask)

        # Encode History Context
        h_history = self.history_encoder(torch.cat([ctx_topic, ctx_narrative], dim=1))

        # Encode Metadata
        h_meta = self.meta_encoder(meta)

        # Concatenate Semantic Features
        s_vector = torch.cat([h_title, h_body, h_history], dim=1)

        # Generate Gate from Metadata
        gate = self.gate_generator(h_meta)

        # Apply Gating
        s_gated = s_vector * gate

        # Final Fusion
        fused = torch.cat([s_gated, h_meta], dim=1)
        logits = self.classifier(fused)

        return logits


# =============================================================================
# TRAINING & INFERENCE
# =============================================================================


def train_nn(
    model, train_loader, val_loader, epochs, patience, lr, weight_decay, device
):
    """
    Training loop with Early Stopping.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Move to device
            meta = batch["meta"].to(device)
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            history = batch["history"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(meta, title, body, history, mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * meta.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                meta = batch["meta"].to(device)
                title = batch["title"].to(device)
                body = batch["body"].to(device)
                history = batch["history"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                logits = model(meta, title, body, history, mask)
                loss = criterion(logits, labels)
                val_loss += loss.item() * meta.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_preds)

        # Print metrics
        # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc}")

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best Validation AUC: {best_auc}")

    return model


def predict_nn(model, loader, device):
    """
    Inference function.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            meta = batch["meta"].to(device)
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            history = batch["history"].to(device)
            mask = batch["mask"].to(device)

            logits = model(meta, title, body, history, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)

    return np.array(preds).flatten()


def run_nn_pipeline(mlp_data):
    """
    Orchestrates the MLP pipeline: Dataset creation, Training, Inference.
    """
    print("Starting Neural Network Pipeline...")
    utils.set_seed()

    # 1. Prepare Datasets
    train_dataset = PizzaDataset(
        mlp_data["meta_train"],
        mlp_data["title_train"],
        mlp_data["body_train"],
        mlp_data["hist_train"],
        mlp_data["mask_train"],
        mlp_data["y_train"],
    )
    val_dataset = PizzaDataset(
        mlp_data["meta_val"],
        mlp_data["title_val"],
        mlp_data["body_val"],
        mlp_data["hist_val"],
        mlp_data["mask_val"],
        mlp_data["y_val"],
    )
    test_dataset = PizzaDataset(
        mlp_data["meta_test"],
        mlp_data["title_test"],
        mlp_data["body_test"],
        mlp_data["hist_test"],
        mlp_data["mask_test"],
        None,
    )

    # 2. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.MLP_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.MLP_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    meta_dim = mlp_data["meta_train"].shape[1]
    device = torch.device(config.DEVICE)

    model = PizzaNet(
        meta_dim=meta_dim,
        embed_dim=config.SBERT_EMBEDDING_DIM,
        hidden_dim=config.MLP_HIDDEN_DIM,
        dropout=config.MLP_DROPOUT,
    ).to(device)

    # 4. Training
    model = train_nn(
        model,
        train_loader,
        val_loader,
        epochs=config.MLP_EPOCHS,
        patience=config.MLP_PATIENCE,
        lr=config.MLP_LEARNING_RATE,
        weight_decay=config.MLP_WEIGHT_DECAY,
        device=device,
    )

    # 5. Inference
    print("Generating predictions...")
    val_preds = predict_nn(model, val_loader, device)
    test_preds = predict_nn(model, test_loader, device)

    # 6. Save Model
    model_path = os.path.join(config.WORKING_DIR, "mlp_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"MLP model saved to {model_path}")

    return val_preds, test_preds, model
