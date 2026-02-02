import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import (
    set_seed,
    save_model_checkpoint,
    load_model_checkpoint,
    ensure_dir,
)

# =========================================================================
# Dataset Wrapper
# =========================================================================


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Dual-Query MLP.
    Wraps the dictionary structure produced by FeatureEngineer.process_stream_b.
    """

    def __init__(self, data_dict, mode="train"):
        self.mode = mode

        # Features
        self.meta = torch.FloatTensor(data_dict["meta"])
        self.title_emb = torch.FloatTensor(data_dict["title_emb"])
        self.body_emb = torch.FloatTensor(data_dict["body_emb"])
        self.hist_emb = torch.FloatTensor(data_dict["hist_emb"])
        self.hist_mask = torch.FloatTensor(data_dict["hist_mask"])

        # Target (only for train/val)
        if "y" in data_dict:
            self.y = torch.FloatTensor(data_dict["y"])
        else:
            self.y = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        sample = {
            "meta": self.meta[idx],
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "hist_emb": self.hist_emb[idx],
            "hist_mask": self.hist_mask[idx],
        }

        if self.y is not None:
            sample["y"] = self.y[idx]

        return sample


# =========================================================================
# Neural Network Architecture
# =========================================================================


class DualQueryAttentionMLP(nn.Module):
    """
    Hybrid Neural Network with Dual-Query Attention and Gated Fusion.

    Architecture:
    1. Semantic Branch: Processes Title and Body embeddings.
    2. History Branch: Dual-Head Attention (Topic & Narrative) querying User History.
    3. Alignment Injection: Computes Cosine Similarity between Query and Context.
    4. Metadata Branch: Generates a 'Credibility Gate' from tabular features.
    5. Fusion: Modulates semantic signals via the Gate.
    """

    def __init__(self, meta_dim):
        super(DualQueryAttentionMLP, self).__init__()

        self.emb_dim = Config.MLP_EMBEDDING_DIM
        self.hidden_dim = Config.MLP_HIDDEN_DIM
        self.dropout_p = Config.MLP_DROPOUT

        # Regularization
        self.dropout = nn.Dropout(self.dropout_p)

        # --- Metadata Branch (Gating Mechanism) ---
        # Projects metadata to a dimension capable of gating the fused semantic vector
        # Fused Vector Size = 4 * Emb_Dim (Title, Body, C_topic, C_narrative) + 2 (Scalars)
        self.fusion_dim = (4 * self.emb_dim) + 2

        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.fusion_dim),
            nn.Sigmoid(),  # Gate values between 0 and 1
        )

        # --- Final Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _attention(self, query, key_value, mask):
        """
        Computes Dot-Product Attention.
        Query: (B, D)
        Key_Value: (B, Seq, D)
        Mask: (B, Seq) - 1 for valid, 0 for padding

        Returns:
            context: (B, D) - Weighted sum of history
            weights: (B, Seq) - Attention weights
        """
        # Expand Query to (B, 1, D)
        query_unsqueezed = query.unsqueeze(1)

        # Scores: (B, 1, D) @ (B, D, Seq) -> (B, 1, Seq)
        # Scale by sqrt(D)
        d_k = query.size(-1)
        scores = torch.bmm(query_unsqueezed, key_value.transpose(1, 2)) / np.sqrt(d_k)
        scores = scores.squeeze(1)  # (B, Seq)

        # Apply Masking (Additive -inf)
        # mask is 1.0 for valid, 0.0 for padding.
        # We want to add -inf where mask is 0.
        extended_mask = (1.0 - mask) * -1e9
        scores = scores + extended_mask

        # Softmax
        weights = F.softmax(scores, dim=-1)  # (B, Seq)

        # Context: (B, 1, Seq) @ (B, Seq, D) -> (B, 1, D)
        context = torch.bmm(weights.unsqueeze(1), key_value).squeeze(1)

        return context

    def _cosine_similarity_feature(self, vec_a, vec_b):
        """
        Computes cosine similarity between two batches of vectors.
        Returns (B, 1)
        """
        # Add epsilon to avoid div by zero
        norm_a = torch.norm(vec_a, dim=1, keepdim=True) + 1e-8
        norm_b = torch.norm(vec_b, dim=1, keepdim=True) + 1e-8

        dot = torch.sum(vec_a * vec_b, dim=1, keepdim=True)
        return dot / (norm_a * norm_b)

    def forward(self, title_emb, body_emb, hist_emb, hist_mask, meta):
        # Apply dropout to embeddings
        title_emb = self.dropout(title_emb)
        body_emb = self.dropout(body_emb)
        hist_emb = self.dropout(hist_emb)

        # --- Dual-Query Attention ---
        # Head A: Topic Context (Query = Title)
        c_topic = self._attention(title_emb, hist_emb, hist_mask)

        # Head B: Narrative Context (Query = Body)
        c_narrative = self._attention(body_emb, hist_emb, hist_mask)

        # --- Alignment Injection ---
        # Compute scalar alignment scores
        s_topic = self._cosine_similarity_feature(title_emb, c_topic)
        s_narrative = self._cosine_similarity_feature(body_emb, c_narrative)

        # Handle case where history is empty (context is zero vector)
        # If mask sum is 0, context is 0. Cosine sim might be noisy or 0.
        # The epsilon in norm handles div/0, resulting in valid (likely small) values.

        # --- Gated Fusion ---
        # Concatenate all semantic signals
        # Dims: D + D + D + D + 1 + 1 = 4D + 2
        semantic_vector = torch.cat(
            [title_emb, body_emb, c_topic, c_narrative, s_topic, s_narrative], dim=1
        )

        # Generate Gate from Metadata
        gate = self.meta_gate(meta)

        # Modulate
        gated_vector = semantic_vector * gate

        # --- Classification ---
        logits = self.classifier(gated_vector)

        return logits


# =========================================================================
# Trainer Class
# =========================================================================


class MLPTrainer:
    """
    Handles training, validation, and inference for the DualQueryAttentionMLP.
    """

    def __init__(self):
        set_seed(Config.RANDOM_STATE)
        self.device = torch.device(Config.DEVICE)
        self.model = None
        self.checkpoint_path = os.path.join(Config.WORKING_DIR, "nn_model.pth")

    def _init_model(self, meta_dim):
        self.model = DualQueryAttentionMLP(meta_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.MLP_LEARNING_RATE,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, train_data, val_data):
        """
        Runs the training loop with Early Stopping.
        """
        # Create Datasets and Loaders
        train_dataset = PizzaDataset(train_data, mode="train")
        val_dataset = PizzaDataset(val_data, mode="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.MLP_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        meta_dim = train_data["meta"].shape[1]
        self._init_model(meta_dim)

        print(f"Starting MLP Training on {self.device}...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        best_auc = 0.0
        patience_counter = 0

        for epoch in range(Config.MLP_EPOCHS):
            # --- Training Step ---
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                # Move to device
                meta = batch["meta"].to(self.device)
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist = batch["hist_emb"].to(self.device)
                mask = batch["hist_mask"].to(self.device)
                y = batch["y"].to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()

                logits = self.model(title, body, hist, mask, meta)
                loss = self.criterion(logits, y)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * y.size(0)

            avg_train_loss = train_loss / len(train_dataset)

            # --- Validation Step ---
            val_auc, val_loss = self._evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # --- Early Stopping & Checkpointing ---
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                save_model_checkpoint(self.model, self.checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.MLP_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc}"
                )
                break

        # Load best model for final state
        self.model = load_model_checkpoint(
            self.model, self.checkpoint_path, device=self.device
        )
        return best_auc

    def _evaluate(self, loader):
        """Helper for evaluation."""
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                meta = batch["meta"].to(self.device)
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist = batch["hist_emb"].to(self.device)
                mask = batch["hist_mask"].to(self.device)
                y = batch["y"].to(self.device).unsqueeze(1)

                logits = self.model(title, body, hist, mask, meta)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * y.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_preds.extend(probs)
                all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.5  # Handle edge cases with single class in batch

        return auc, avg_loss

    def predict_proba(self, test_data):
        """
        Generates predictions for the test set.
        """
        if self.model is None:
            # Try to load if not in memory (e.g. inference only run)
            meta_dim = test_data["meta"].shape[1]
            self._init_model(meta_dim)
            try:
                self.model = load_model_checkpoint(
                    self.model, self.checkpoint_path, device=self.device
                )
            except FileNotFoundError:
                raise RuntimeError("Model not trained and no checkpoint found.")

        test_dataset = PizzaDataset(test_data, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.MLP_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        self.model.eval()
        all_preds = []

        print("Running MLP Inference...")
        with torch.no_grad():
            for batch in test_loader:
                meta = batch["meta"].to(self.device)
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist = batch["hist_emb"].to(self.device)
                mask = batch["hist_mask"].to(self.device)

                logits = self.model(title, body, hist, mask, meta)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)

        return np.array(all_preds).flatten()

    def save(self, path):
        # Wrapper for saving artifact if needed (though checkpoint is already saved)
        pass

    def load(self, path):
        pass
