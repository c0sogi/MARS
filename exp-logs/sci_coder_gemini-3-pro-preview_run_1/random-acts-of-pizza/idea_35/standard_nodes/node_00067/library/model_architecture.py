import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Random Acts of Pizza task.
    Handles multi-modal input: Title/Body embeddings, History sequences, and Metadata.
    """

    def __init__(self, title, body, history, meta, y=None):
        """
        Args:
            title (np.ndarray): SBERT embeddings of request titles (N, D).
            body (np.ndarray): SBERT embeddings of request bodies (N, D).
            history (np.ndarray): SBERT embeddings of user history (N, L, D).
            meta (np.ndarray): Preprocessed metadata features (N, M).
            y (np.ndarray, optional): Target labels (N,).
        """
        self.title = torch.FloatTensor(title)
        self.body = torch.FloatTensor(body)
        self.history = torch.FloatTensor(history)
        self.meta = torch.FloatTensor(meta)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        sample = {
            "title": self.title[idx],
            "body": self.body[idx],
            "history": self.history[idx],
            "meta": self.meta[idx],
        }
        if self.y is not None:
            sample["y"] = self.y[idx]
        return sample


class DualQueryAttention(nn.Module):
    """
    Computes attention context between a query (Title/Body) and user history.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.scale = np.sqrt(embed_dim)

    def forward(self, query, history):
        """
        Args:
            query: (B, D)
            history: (B, L, D)
        Returns:
            context: (B, D)
        """
        # Q: (B, 1, D), K: (B, L, D)
        Q = query.unsqueeze(1)
        K = history
        V = history

        # Scores: (B, 1, L)
        # Calculate dot product attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Masking: Assume 0-vectors in history are padding
        # Sum absolute values across embedding dim to find padded steps
        # shape: (B, L) -> (B, 1, L)
        mask = (history.abs().sum(dim=2) == 0).unsqueeze(1)

        # Apply mask (set padded positions to -infinity)
        scores = scores.masked_fill(mask, -1e9)

        # Attention Weights: (B, 1, L)
        attn_weights = F.softmax(scores, dim=-1)

        # Context: (B, 1, D) -> (B, D)
        context = torch.bmm(attn_weights, V).squeeze(1)
        return context


class DualQueryMLP(nn.Module):
    """
    Dropout-Stabilized Dual-Query MLP.
    Features:
    - Dual-Query Attention (Title->History, Body->History)
    - Metadata-driven Gated Fusion (Credibility Gate)
    - Dropout-only regularization (No Batch Norm)
    """

    def __init__(self, emb_dim, meta_dim, hidden_dim, dropout=0.5):
        super().__init__()

        # Components
        self.attention_module = DualQueryAttention(emb_dim)

        # Fusion Dimension: Title + Body + Ctx_Title + Ctx_Body
        self.fusion_dim = 4 * emb_dim

        # Gating Network (Metadata -> Gate)
        # Modulates the semantic vector based on user credibility/metadata
        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.fusion_dim),
            nn.Sigmoid(),
        )

        # Main Classifier
        # Standard MLP structure with Dropout, no Batch Norm
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, title, body, history, meta):
        """
        Args:
            title: (B, D)
            body: (B, D)
            history: (B, L, D)
            meta: (B, M)
        Returns:
            logits: (B,)
        """
        # 1. Dual Query Attention
        # Extract context from history relevant to title and body
        ctx_title = self.attention_module(title, history)
        ctx_body = self.attention_module(body, history)

        # 2. Concatenate Semantics
        # Combine all semantic signals: (B, 4*D)
        semantic_vec = torch.cat([title, body, ctx_title, ctx_body], dim=1)
        semantic_vec = self.dropout(semantic_vec)

        # 3. Gated Fusion
        # Generate gate from metadata (including global alignment scalars)
        gate = self.meta_gate(meta)

        # Modulate semantic vector (Element-wise multiplication)
        fused = semantic_vec * gate

        # 4. Classification
        logits = self.classifier(fused)

        # Return shape (B,)
        return logits.squeeze(1)
