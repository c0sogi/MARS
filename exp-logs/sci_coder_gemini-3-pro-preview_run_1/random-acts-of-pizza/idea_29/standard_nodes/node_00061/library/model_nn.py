import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import NN_PARAMS, CACHE_DIR, EMBEDDING_DIM
from library.utils import get_device, set_seed


class DualQueryAttention(nn.Module):
    """
    Computes Dot-Product Attention between a Query vector and a sequence of Key vectors.
    Includes explicit additive masking for padding.
    """

    def __init__(self, embed_dim, dropout=0.1):
        super(DualQueryAttention, self).__init__()
        self.scale = embed_dim**-0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, keys):
        """
        Args:
            query: (Batch, Dim)
            keys: (Batch, Seq_Len, Dim)
        Returns:
            context: (Batch, Dim)
        """
        # Calculate scores: (Batch, Seq_Len)
        # query.unsqueeze(2) -> (Batch, Dim, 1)
        # bmm with keys -> (Batch, Seq_Len)
        # Alternatively: (Q * K).sum(dim=-1)

        # Expand query to (Batch, 1, Dim) for broadcasting or matmul
        # keys is (Batch, Seq, Dim)
        # We want (Batch, Seq)
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        scores = scores * self.scale

        # Create mask for padding (assuming zero-padding in keys)
        # If a key vector is all zeros, it's padding.
        # Check L2 norm or sum of abs
        is_padding = keys.abs().sum(dim=2) == 0  # (Batch, Seq)

        # Additive Masking
        mask_value = -1e9
        scores = scores.masked_fill(is_padding, mask_value)

        # Softmax
        attn_weights = F.softmax(scores, dim=1)
        attn_weights = self.dropout(attn_weights)

        # Weighted Sum: (Batch, 1, Seq) @ (Batch, Seq, Dim) -> (Batch, 1, Dim)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)

        return context


class PizzaNetwork(nn.Module):
    """
    Alignment-Injected Dual-Query MLP.
    """

    def __init__(self, metadata_dim, hidden_dim=256, dropout_rate=0.3):
        super(PizzaNetwork, self).__init__()

        self.embedding_dim = EMBEDDING_DIM

        # Dropout for raw embeddings
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Attention Modules
        self.attn_topic = DualQueryAttention(EMBEDDING_DIM, dropout=dropout_rate)
        self.attn_narrative = DualQueryAttention(EMBEDDING_DIM, dropout=dropout_rate)

        # Metadata Encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Fusion Dimensions
        # Title + Body + C_topic + C_narrative + S_topic + S_narrative
        self.fusion_dim = (EMBEDDING_DIM * 4) + 2

        # Gating Mechanism: Metadata -> Gate
        self.gate_layer = nn.Linear(hidden_dim, self.fusion_dim)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, title_emb, body_emb, history_emb, metadata):
        # 1. Process Raw Embeddings
        t_emb = self.emb_dropout(title_emb)
        b_emb = self.emb_dropout(body_emb)
        h_emb = self.emb_dropout(history_emb)

        # 2. Dual-Query Attention
        # Head A: Topic Context (Query=Title)
        c_topic = self.attn_topic(t_emb, h_emb)

        # Head B: Narrative Context (Query=Body)
        c_narrative = self.attn_narrative(b_emb, h_emb)

        # 3. Alignment Injection (Cosine Similarity)
        # Add epsilon to avoid div by zero
        eps = 1e-8
        s_topic = F.cosine_similarity(t_emb, c_topic, dim=1, eps=eps).unsqueeze(1)
        s_narrative = F.cosine_similarity(b_emb, c_narrative, dim=1, eps=eps).unsqueeze(
            1
        )

        # 4. Concatenation
        combined_semantics = torch.cat(
            [t_emb, b_emb, c_topic, c_narrative, s_topic, s_narrative], dim=1
        )

        # 5. Metadata Gating
        meta_feat = self.meta_encoder(metadata)
        gate = torch.sigmoid(self.gate_layer(meta_feat))

        # Gated Fusion
        gated_features = combined_semantics * gate

        # 6. Classification
        logits = self.classifier(gated_features)

        return logits


class PizzaNeuralNet:
    """
    Wrapper class for training and inference of the PizzaNetwork.
    """

    def __init__(self, metadata_dim):
        self.device = get_device()
        self.model = PizzaNetwork(
            metadata_dim=metadata_dim,
            hidden_dim=NN_PARAMS["hidden_dim"],
            dropout_rate=NN_PARAMS["dropout_rate"],
        ).to(self.device)

        self.model_path = os.path.join(CACHE_DIR, "nn_model.pth")

    def train(self, train_loader, val_loader):
        """
        Trains the neural network with Early Stopping.
        """
        print("Starting Neural Network training...")

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=NN_PARAMS["learning_rate"],
            weight_decay=NN_PARAMS["weight_decay"],
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(NN_PARAMS["epochs"]):
            self.model.train()
            train_losses = []

            for batch in train_loader:
                # Move data to device
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                history = batch["history_emb"].to(self.device)
                meta = batch["metadata"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(title, body, history, meta)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation
            val_auc, val_loss = self.evaluate(val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{NN_PARAMS['epochs']} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val AUC: {val_auc}"
            )  # Full precision

            # Early Stopping Check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= NN_PARAMS["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_val_auc

    def evaluate(self, loader, criterion=None):
        """
        Evaluates the model on a given loader.
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        losses = []

        with torch.no_grad():
            for batch in loader:
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                history = batch["history_emb"].to(self.device)
                meta = batch["metadata"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)

                logits = self.model(title, body, history, meta)
                probs = torch.sigmoid(logits)

                if criterion:
                    loss = criterion(logits, labels)
                    losses.append(loss.item())

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs).flatten()
        all_labels = np.array(all_labels).flatten()

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # Handle edge case with single class in batch

        avg_loss = np.mean(losses) if losses else 0.0

        return auc, avg_loss

    def predict_proba(self, loader):
        """
        Generates predictions for inference.
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                history = batch["history_emb"].to(self.device)
                meta = batch["metadata"].to(self.device)

                logits = self.model(title, body, history, meta)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs).flatten()

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"NN model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            print(f"NN model loaded from {self.model_path}")
            return True
        return False
