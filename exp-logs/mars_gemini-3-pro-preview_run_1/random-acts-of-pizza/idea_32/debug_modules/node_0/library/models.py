import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from library.config import Config


class DualQueryGatedMLP(nn.Module):
    """
    Dual-Query Alignment-Gated MLP (Dropout-Only).

    This neural network models the complex interaction between the request content
    (Title, Body) and the user's history using a dual-query attention mechanism.
    It incorporates metadata via a Gated Fusion mechanism to modulate the semantic
    signals based on user credibility/activity metrics.

    Architecture:
    1. Inputs: Raw SBERT embeddings (Title, Body, History) + Scaled Metadata.
    2. Attention: Dot-Product Attention with explicit Alignment Injection.
    3. Fusion: Metadata generates a Sigmoid gate applied to the semantic vector.
    4. Regularization: Dropout only (No Batch Normalization).
    """

    def __init__(
        self,
        embedding_dim=Config.SBERT_EMBEDDING_DIM,
        metadata_dim=len(Config.NUMERIC_COLS),
        hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout_emb=Config.MLP_DROPOUT_EMB,
        dropout_dense=Config.MLP_DROPOUT_DENSE,
    ):
        super(DualQueryGatedMLP, self).__init__()

        self.embedding_dim = embedding_dim

        # --- Semantic Processing Layers ---
        # Input construction:
        # Title(D) + Body(D) + Context_Title(D) + Context_Body(D) + Align_Title(1) + Align_Body(1)
        semantic_input_dim = (4 * embedding_dim) + 2

        self.semantic_projection = nn.Sequential(
            nn.Linear(semantic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
        )

        # --- Metadata Processing & Gating ---
        # Metadata processes to a hidden representation, which then generates a gate
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
        )

        # The gate generator produces a value in (0, 1) to modulate the semantic vector
        self.gate_generator = nn.Linear(hidden_dim, hidden_dim)

        # --- Final Classifier ---
        # Takes the fused vector (Semantic * Gate)
        self.classifier = nn.Linear(hidden_dim, 1)

        # --- Input Regularization ---
        self.dropout_input = nn.Dropout(dropout_emb)

    def attention(self, query, key, value, mask=None):
        """
        Computes Scaled Dot-Product Attention and extracts Alignment Scalars.

        Args:
            query: Tensor (B, D)
            key: Tensor (B, L, D)
            value: Tensor (B, L, D)
            mask: BoolTensor (B, L) - True indicates padding position.

        Returns:
            context: Tensor (B, D) - Weighted sum of values.
            alignment: Tensor (B, 1) - Dot product between Query and Context.
        """
        # Expand query for batch matrix multiplication: (B, 1, D)
        q_unsqueezed = query.unsqueeze(1)

        # Compute Scores: (B, 1, D) @ (B, D, L) -> (B, 1, L)
        # Transpose key to (B, D, L)
        scores = torch.bmm(q_unsqueezed, key.transpose(1, 2))

        # Scale scores
        scores = scores / np.sqrt(self.embedding_dim)

        # Apply Masking (Additive -inf)
        if mask is not None:
            # mask is (B, L), expand to (B, 1, L)
            mask_expanded = mask.unsqueeze(1)
            scores = scores.masked_fill(mask_expanded, -1e9)

        # Compute Attention Weights
        weights = F.softmax(scores, dim=-1)  # (B, 1, L)

        # Compute Context: (B, 1, L) @ (B, L, D) -> (B, 1, D)
        context = torch.bmm(weights, value)
        context = context.squeeze(1)  # (B, D)

        # Compute Alignment Scalar: Dot product between Query and Context
        # Represents how well the retrieved context aligns with the query
        # (B, D) * (B, D) -> sum(dim=1) -> (B, 1)
        alignment = (query * context).sum(dim=1, keepdim=True)

        return context, alignment

    def forward(
        self, title_emb, body_emb, history_emb, metadata, history_padding_mask=None
    ):
        """
        Forward pass of the network.
        """
        # Apply high dropout to input embeddings (Regularization)
        t_emb = self.dropout_input(title_emb)
        b_emb = self.dropout_input(body_emb)
        h_emb = self.dropout_input(history_emb)

        # --- Branch 3: Dual-Query History Attention ---
        # Head A: Topic Context (Query = Title)
        ctx_title, align_title = self.attention(
            t_emb, h_emb, h_emb, history_padding_mask
        )

        # Head B: Narrative Context (Query = Body)
        ctx_body, align_body = self.attention(b_emb, h_emb, h_emb, history_padding_mask)

        # --- Concatenation ---
        # Combine all semantic signals
        semantic_features = torch.cat(
            [t_emb, b_emb, ctx_title, ctx_body, align_title, align_body], dim=1
        )

        # Project to hidden space
        semantic_vec = self.semantic_projection(semantic_features)  # (B, Hidden)

        # --- Branch 4: Metadata & Gated Fusion ---
        # Process metadata
        meta_vec = self.metadata_mlp(metadata)  # (B, Hidden)

        # Generate Credibility Gate
        gate = torch.sigmoid(
            self.gate_generator(meta_vec)
        )  # (B, Hidden) in range (0,1)

        # Fused Vector: Semantic content modulated by Metadata credibility
        fused_vec = semantic_vec * gate

        # --- Output ---
        logits = self.classifier(fused_vec)
        return logits


class RFWrapper:
    """
    Wrapper for Scikit-Learn's RandomForestClassifier.
    Encapsulates configuration, training, prediction, and persistence.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            class_weight=Config.RF_CLASS_WEIGHT,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RANDOM_SEED,
            verbose=0,
        )
        self.is_fitted = False

    def fit(self, X, y):
        """
        Trains the Random Forest model.
        Args:
            X (sparse matrix or array): Feature matrix.
            y (array): Target labels.
        """
        print(
            f"Training Random Forest with {X.shape[0]} samples and {X.shape[1]} features..."
        )
        self.model.fit(X, y)
        self.is_fitted = True
        print("Random Forest training complete.")

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        # Return probability of positive class (index 1)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        """Saves the model using joblib."""
        print(f"Saving Random Forest model to {path}...")
        joblib.dump(self.model, path)

    def load(self, path):
        """Loads the model using joblib."""
        if os.path.exists(path):
            print(f"Loading Random Forest model from {path}...")
            self.model = joblib.load(path)
            self.is_fitted = True
        else:
            print(f"Model file {path} not found.")
