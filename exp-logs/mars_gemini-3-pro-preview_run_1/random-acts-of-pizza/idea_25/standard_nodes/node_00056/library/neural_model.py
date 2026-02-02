import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
import time
from library.utils import get_device


class MaskedAttentionLayer(nn.Module):
    """
    Computes dot-product attention between a Query (Request) and Keys (History)
    with explicit masking for padding tokens.
    """

    def __init__(self, input_dim):
        super(MaskedAttentionLayer, self).__init__()
        self.scale = input_dim**-0.5

    def forward(self, query, keys, mask):
        """
        Args:
            query: (Batch, Dim) - Request embedding
            keys: (Batch, Seq_Len, Dim) - History embeddings
            mask: (Batch, Seq_Len) - 1 for valid token, 0 for padding

        Returns:
            context: (Batch, Dim) - Weighted sum of keys
            weights: (Batch, Seq_Len) - Attention weights
        """
        # Expand query to (Batch, 1, Dim)
        query_unsqueezed = query.unsqueeze(1)

        # Compute scores: (Batch, 1, Seq_Len)
        # Q * K^T
        scores = torch.bmm(query_unsqueezed, keys.transpose(1, 2))
        scores = scores * self.scale

        # Apply additive mask
        # mask is (Batch, Seq_Len), expand to (Batch, 1, Seq_Len)
        mask_expanded = mask.unsqueeze(1)

        # Fill padding positions with -infinity so softmax becomes 0
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Softmax over sequence dimension
        weights = F.softmax(scores, dim=-1)

        # Compute context: (Batch, 1, Seq_Len) * (Batch, Seq_Len, Dim) -> (Batch, 1, Dim)
        context = torch.bmm(weights, keys)

        # Squeeze back to (Batch, Dim)
        context = context.squeeze(1)

        return context, weights.squeeze(1)


class GatedPizzaNetwork(nn.Module):
    """
    Hybrid Neural Network with:
    1. Request SBERT Branch
    2. Masked Attention History Branch
    3. Metadata Credibility Gating
    """

    def __init__(self, text_dim=384, meta_dim=10, hidden_dim=128, dropout_rate=0.3):
        super(GatedPizzaNetwork, self).__init__()

        self.text_dim = text_dim

        # --- Branch 1 & 2: Text Interaction ---
        self.attention = MaskedAttentionLayer(text_dim)
        self.text_dropout = nn.Dropout(dropout_rate)

        # The semantic vector will be Concatenation(Request, History_Context)
        self.semantic_dim = text_dim * 2

        # --- Branch 3: Metadata Gating ---
        # Metadata -> Hidden -> Gate (size of semantic vector)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, self.semantic_dim),
            nn.Sigmoid(),  # Gate values between 0 and 1
        )

        # --- Final Prediction Head ---
        # Gated Semantic Vector -> Prediction
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, request_emb, history_seq, history_mask, metadata):
        """
        Args:
            request_emb: (B, 384)
            history_seq: (B, 20, 384)
            history_mask: (B, 20)
            metadata: (B, Meta_Dim)
        """
        # 1. Apply Dropout to raw embeddings (Regularization)
        req_emb = self.text_dropout(request_emb)
        hist_seq = self.text_dropout(history_seq)

        # 2. Compute History Context via Attention
        # context: (B, 384)
        context, _ = self.attention(req_emb, hist_seq, history_mask)

        # 3. Construct Semantic Vector
        # (B, 768)
        semantic_vector = torch.cat([req_emb, context], dim=1)

        # 4. Compute Credibility Gate from Metadata
        # (B, 768)
        gate = self.meta_mlp(metadata)

        # 5. Apply Gating (Element-wise multiplication)
        # "Modulated" semantic vector
        gated_vector = semantic_vector * gate

        # 6. Final Classification
        logits = self.classifier(gated_vector)

        return logits


def train_neural_model(train_loader, val_loader, input_dims, config=None):
    """
    Trains the GatedPizzaNetwork with Early Stopping.

    Args:
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        input_dims: dict containing 'text_dim' and 'meta_dim'
        config: dict containing hyperparameters (lr, epochs, patience, etc.)

    Returns:
        model: The best trained model (loaded from state_dict)
        history: dict of training logs
    """
    if config is None:
        config = {
            "lr": 1e-4,
            "epochs": 50,
            "patience": 15,
            "hidden_dim": 256,
            "dropout": 0.3,
            "weight_decay": 1e-4,
        }

    device = get_device()
    print(f"Training Neural Model on {device}")

    # Initialize Model
    model = GatedPizzaNetwork(
        text_dim=input_dims["text_dim"],
        meta_dim=input_dims["meta_dim"],
        hidden_dim=config["hidden_dim"],
        dropout_rate=config["dropout"],
    ).to(device)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    best_val_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_auc": []}

    for epoch in range(config["epochs"]):
        # --- Training ---
        model.train()
        train_losses = []

        for batch in train_loader:
            # Move data to device
            req = batch["request_emb"].to(device)
            hist = batch["history_seq"].to(device)
            mask = batch["history_mask"].to(device)
            meta = batch["metadata"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(req, hist, mask, meta)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                req = batch["request_emb"].to(device)
                hist = batch["history_seq"].to(device)
                mask = batch["history_mask"].to(device)
                meta = batch["metadata"].to(device)
                labels = batch["label"].to(device)

                outputs = model(req, hist, mask, meta)
                probs = torch.sigmoid(outputs).cpu().numpy()

                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # --- Early Stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Best Validation AUC: {best_val_auc:.10f}")

    return model, history
