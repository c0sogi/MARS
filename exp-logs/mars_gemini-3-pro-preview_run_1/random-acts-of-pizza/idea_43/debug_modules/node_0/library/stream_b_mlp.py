import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, Optional

from library.config import Config
from library.utils import set_seed, get_device
from library.feature_engineering import FeatureProcessor


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Hybrid Ensemble Stream B.
    Pre-loads all data to the specified device (GPU/CPU) for efficiency,
    given the dataset size fits in memory.
    """

    def __init__(self, data_dict: Dict[str, np.ndarray], device: torch.device):
        self.device = device

        # Unified Credibility Block Components
        self.meta = torch.tensor(data_dict["X_meta"], dtype=torch.float32).to(device)
        self.topk = torch.tensor(data_dict["X_topk"], dtype=torch.float32).to(device)
        self.centroids = torch.tensor(
            data_dict["history_centroids"], dtype=torch.float32
        ).to(device)

        # Semantic Components
        self.title_emb = torch.tensor(data_dict["emb_title"], dtype=torch.float32).to(
            device
        )
        self.body_emb = torch.tensor(data_dict["emb_body"], dtype=torch.float32).to(
            device
        )
        self.history_seq = torch.tensor(
            data_dict["history_sequences"], dtype=torch.float32
        ).to(device)
        self.consistency = torch.tensor(
            data_dict["consistency"], dtype=torch.float32
        ).to(device)

        # Target
        if "y" in data_dict:
            self.y = torch.tensor(data_dict["y"], dtype=torch.float32).to(device)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            "meta": self.meta[idx],
            "topk": self.topk[idx],
            "centroid": self.centroids[idx],
            "title": self.title_emb[idx],
            "body": self.body_emb[idx],
            "history": self.history_seq[idx],
            "consistency": self.consistency[idx],
        }
        if self.y is not None:
            batch["y"] = self.y[idx]
        return batch


class DualQueryAttention(nn.Module):
    """
    Computes attention over user history sequences using a query vector.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.scale = embed_dim**-0.5

    def forward(self, query: torch.Tensor, key_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (Batch, Dim)
            key_values: (Batch, Seq_Len, Dim)
        Returns:
            context: (Batch, Dim)
        """
        # Create mask for padding (assuming zero-padding implies all zeros)
        # Mask is True where sequence is NOT padding
        mask = key_values.abs().sum(dim=-1) > 1e-6  # (Batch, Seq_Len)

        # Q: (B, 1, D), K: (B, L, D) -> Scores: (B, 1, L)
        q = query.unsqueeze(1)
        scores = torch.bmm(q, key_values.transpose(1, 2)) * self.scale

        # Apply mask: set padding positions to -inf
        # scores is (B, 1, L), mask is (B, L) -> unsqueeze mask to (B, 1, L)
        mask_expanded = mask.unsqueeze(1)
        scores = scores.masked_fill(~mask_expanded, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum: (B, 1, L) x (B, L, D) -> (B, 1, D)
        context = torch.bmm(attn_weights, key_values)

        return context.squeeze(1)


class CredibilityGatedMLP(nn.Module):
    """
    Unified Credibility-Gated MLP.
    Combines Dual-Query Attention on history with a Unified Credibility Block
    via a Skip-Gated Fusion mechanism.
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.emb_dim = Config.EMBEDDING_DIM
        self.meta_dim = 13  # From FeatureProcessor
        self.topk_dim = Config.TOP_K_SUBREDDITS
        self.consistency_dim = 2

        # Unified Credibility Block Dimension
        # Meta (13) + TopK (50) + Centroid (384)
        self.credibility_dim = self.meta_dim + self.topk_dim + self.emb_dim

        # Semantic Vector Dimension
        # Title (384) + Body (384) + Attn_Title (384) + Attn_Body (384) + Consistency (2)
        self.semantic_dim = (self.emb_dim * 4) + self.consistency_dim

        # Layers
        self.attention = DualQueryAttention(self.emb_dim)

        # Gating Network: Credibility -> Gate (size of Semantic)
        self.gate_fc = nn.Linear(self.credibility_dim, self.semantic_dim)

        # Final Fusion: Gated Semantic + Credibility
        self.fusion_dim = self.semantic_dim + self.credibility_dim

        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT_DENSE),
            nn.Linear(self.fusion_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_DENSE),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM // 2, 1),
        )

        self.dropout_emb = nn.Dropout(Config.DROPOUT_EMB)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Construct Unified Credibility Block
        # (B, 13 + 50 + 384)
        credibility_block = torch.cat(
            [batch["meta"], batch["topk"], batch["centroid"]], dim=1
        )

        # 2. Semantic Processing
        title = self.dropout_emb(batch["title"])
        body = self.dropout_emb(batch["body"])
        history = self.dropout_emb(batch["history"])

        # Dual-Query Attention
        context_title = self.attention(title, history)
        context_body = self.attention(body, history)

        # Construct Semantic Vector
        # (B, 384*4 + 2)
        semantic_vector = torch.cat(
            [title, body, context_title, context_body, batch["consistency"]], dim=1
        )

        # 3. Skip-Gated Fusion
        # Compute Gate
        gate = torch.sigmoid(self.gate_fc(credibility_block))

        # Modulate Semantic Vector
        gated_semantic = semantic_vector * gate

        # Final Concatenation (Skip Connection of Credibility Block)
        fused = torch.cat([gated_semantic, credibility_block], dim=1)

        # 4. Classification
        logits = self.classifier(fused)
        return logits.squeeze(1)


class MLPPipeline:
    """
    Stream B Pipeline: Orchestrates Feature Processing, Dataset Creation,
    Model Training (with Early Stopping), and Inference.
    """

    def __init__(self):
        set_seed(Config.SEED)
        self.device = get_device()
        self.model = CredibilityGatedMLP().to(self.device)
        self.feature_processor = FeatureProcessor()

    def train_mlp(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Handles the training loop with AdamW, Dropout, and Early Stopping.
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_state = None
        patience_counter = 0

        print(f"Starting MLP Training on {self.device}...")

        for epoch in range(Config.NUM_EPOCHS):
            self.model.train()
            train_losses = []

            for batch in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch)
                loss = criterion(logits, batch["y"])
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    logits = self.model(batch)
                    probs = torch.sigmoid(logits)
                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(batch["y"].cpu().numpy())

            val_auc = roc_auc_score(val_targets, val_preds)
            avg_train_loss = np.mean(train_losses)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val AUC: {val_auc:.10f}"
            )

            # Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model with Val AUC: {best_auc:.10f}")

    def predict(self, loader: DataLoader) -> np.ndarray:
        """
        Generates probabilities for a dataset.
        """
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                preds.extend(probs.cpu().numpy())
        return np.array(preds)

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Executes the full pipeline.
        """
        # 1. Feature Engineering
        data_dicts = self.feature_processor.process_data(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )

        # 2. Dataset & DataLoader
        train_dataset = PizzaDataset(data_dicts["train"], self.device)
        val_dataset = PizzaDataset(data_dicts["val"], self.device)
        test_dataset = PizzaDataset(data_dicts["test"], self.device)

        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # 3. Train
        self.train_mlp(train_loader, val_loader)

        # 4. Inference
        print("Generating validation predictions...")
        val_preds = self.predict(val_loader)

        print("Generating test predictions...")
        test_preds = self.predict(test_loader)

        return val_preds, test_preds
