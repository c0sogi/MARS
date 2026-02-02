import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import os

from library.config import Config


class GatedAttentionMLP(nn.Module):
    """
    Neural Network with Direct-Attention and Credibility Gating.

    Branches:
    1. Request Branch: Raw SBERT embedding.
    2. History Branch: Dot-Product Attention (Query=Request, Key=History).
    3. Metadata Branch: MLP generating a gate.

    Fusion:
    (Request + History_Context) * Sigmoid(Metadata_Gate)
    """

    def __init__(self, metadata_dim, hidden_dim=256, embedding_dim=384, dropout=0.3):
        super(GatedAttentionMLP, self).__init__()

        self.embedding_dim = embedding_dim

        # Metadata Encoder
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Gating Mechanism: Projects metadata to semantic dimension (Request + Context)
        # Semantic dim = embedding_dim (Request) + embedding_dim (History Context) = 2 * 384
        self.semantic_dim = embedding_dim * 2
        self.gate_proj = nn.Linear(hidden_dim, self.semantic_dim)

        # Final Classification Head
        # Input: Gated Semantics + Metadata Latent
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.semantic_dim + hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, request_emb, history_emb, metadata):
        """
        Args:
            request_emb: (B, 384)
            history_emb: (B, Max_Len, 384)
            metadata: (B, Meta_Dim)
        """
        batch_size = request_emb.size(0)

        # --- 1. History Attention ---
        # Query: Request (B, 1, 384)
        query = request_emb.unsqueeze(1)

        # Keys: History (B, L, 384)
        keys = history_emb

        # Scores: (B, 1, L)
        scores = torch.bmm(query, keys.transpose(1, 2))

        # Scale scores
        scores = scores / (self.embedding_dim**0.5)

        # Masking: Identify padding (all-zero vectors) in history
        # history_emb is (B, L, 384). Sum abs across dim 2.
        # If sum is effectively 0, it's padding.
        mask = history_emb.abs().sum(dim=2) > 1e-9  # (B, L)
        mask = mask.unsqueeze(1)  # (B, 1, L)

        # Apply mask: fill False (padding) with -inf
        scores = scores.masked_fill(~mask, -1e9)

        # Weights: (B, 1, L)
        attn_weights = torch.softmax(scores, dim=-1)

        # Context: (B, 1, 384) -> (B, 384)
        context = torch.bmm(attn_weights, keys).squeeze(1)

        # --- 2. Semantic Aggregation ---
        # Concatenate Request and Attended History
        semantic_vec = torch.cat([request_emb, context], dim=1)  # (B, 768)

        # --- 3. Metadata Gating ---
        # Encode metadata
        meta_latent = self.meta_mlp(metadata)  # (B, Hidden)

        # Generate Gate
        gate = torch.sigmoid(self.gate_proj(meta_latent))  # (B, 768)

        # Apply Gate
        gated_semantics = semantic_vec * gate

        # --- 4. Classification ---
        # Combine gated semantics with metadata latent for final decision
        combined = torch.cat([gated_semantics, meta_latent], dim=1)

        logits = self.classifier(combined)

        return logits


class NeuralNetTrainer:
    """
    Wrapper for training and inference of the GatedAttentionMLP.
    """

    def __init__(self, input_dims):
        self.device = torch.device(Config.DEVICE)
        self.set_seed(Config.RANDOM_SEED)

        self.model = GatedAttentionMLP(
            metadata_dim=input_dims["metadata"],
            hidden_dim=Config.MLP_HIDDEN_DIM,
            embedding_dim=Config.SBERT_EMBEDDING_DIM,
            dropout=Config.MLP_DROPOUT,
        ).to(self.device)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def _prepare_loader(self, data_dict, shuffle=False):
        """
        Converts numpy dict to DataLoader.
        Expected keys: 'metadata', 'request_emb', 'history_emb', 'y' (optional)
        """
        meta = torch.tensor(data_dict["metadata"], dtype=torch.float32)
        req = torch.tensor(data_dict["request_emb"], dtype=torch.float32)
        hist = torch.tensor(data_dict["history_emb"], dtype=torch.float32)

        if "y" in data_dict:
            y = torch.tensor(data_dict["y"], dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(req, hist, meta, y)
        else:
            dataset = TensorDataset(req, hist, meta)

        return DataLoader(
            dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing overhead for simple tensors
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def train(self, train_data, val_data):
        train_loader = self._prepare_loader(train_data, shuffle=True)
        val_loader = self._prepare_loader(val_data, shuffle=False)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.MLP_LEARNING_RATE,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None

        print(f"Starting MLP Training on {self.device}...")

        for epoch in range(Config.MLP_EPOCHS):
            self.model.train()
            train_loss = 0.0

            for req, hist, meta, y in train_loader:
                req, hist, meta, y = (
                    req.to(self.device),
                    hist.to(self.device),
                    meta.to(self.device),
                    y.to(self.device),
                )

                optimizer.zero_grad()
                logits = self.model(req, hist, meta)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * req.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            val_auc, val_loss = self.evaluate(val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
            )

            # Early Stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.MLP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_val_auc

    def evaluate(self, dataloader, criterion=None):
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 4:
                    req, hist, meta, y = batch
                    req, hist, meta, y = (
                        req.to(self.device),
                        hist.to(self.device),
                        meta.to(self.device),
                        y.to(self.device),
                    )

                    logits = self.model(req, hist, meta)
                    probs = torch.sigmoid(logits)

                    if criterion:
                        loss = criterion(logits, y)
                        total_loss += loss.item() * req.size(0)

                    all_preds.append(probs.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
                else:
                    # Inference mode (no targets)
                    req, hist, meta = batch
                    req, hist, meta = (
                        req.to(self.device),
                        hist.to(self.device),
                        meta.to(self.device),
                    )
                    logits = self.model(req, hist, meta)
                    probs = torch.sigmoid(logits)
                    all_preds.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        if all_targets:
            all_targets = np.concatenate(all_targets)
            auc = roc_auc_score(all_targets, all_preds)
            avg_loss = total_loss / len(dataloader.dataset) if criterion else 0.0
            return auc, avg_loss
        else:
            return all_preds

    def predict(self, test_data):
        loader = self._prepare_loader(test_data, shuffle=False)
        preds = self.evaluate(loader)
        return preds.flatten()
