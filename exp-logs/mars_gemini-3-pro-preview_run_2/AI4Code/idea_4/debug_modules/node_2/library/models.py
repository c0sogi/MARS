import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoConfig
from library.config import Config


class TransformerRanker(nn.Module):
    """
    Transformer-based regression model for ranking notebook cells.
    Uses a pre-trained backbone (e.g., DistilRoBERTa) and a linear regression head.
    The model predicts a scalar score (rank) for each input sequence.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        """
        Args:
            model_name (str): Name of the HuggingFace model to use.
            pretrained (bool): Whether to load pre-trained weights.
        """
        super(TransformerRanker, self).__init__()
        self.model_name = model_name

        if pretrained:
            self.config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(self.config)

        # Regression head: projects [CLS] embedding to a single scalar (rank)
        # The hidden size depends on the backbone (e.g., 768 for base models)
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.

        Returns:
            torch.Tensor: Predicted ranks of shape (batch_size,).
        """
        # Pass through backbone
        # outputs is a BaseModelOutputWithPoolingAndCrossAttentions
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token representation (first token in the sequence)
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Project to scalar
        logits = self.regressor(cls_embedding)

        # Squeeze the last dimension to get shape (batch_size,)
        return logits.squeeze(-1)

    def save(self, path=None):
        """Saves the model state dict to disk."""
        if path is None:
            path = Config.TRANSFORMER_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Transformer model saved to {path}")

    def load(self, path=None, device=Config.DEVICE):
        """Loads the model state dict from disk."""
        if path is None:
            path = Config.TRANSFORMER_MODEL_PATH

        if os.path.exists(path):
            # Map location ensures weights are loaded to the correct device
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            print(f"Transformer model loaded from {path}")
        else:
            print(
                f"Warning: No model found at {path}. Model remains initialized with random/pretrained weights."
            )
        return self


class RidgeRanker:
    """
    Wrapper for Ridge Regression model using Scikit-Learn.
    Handles sparse feature inputs for the global lexical stream.
    """

    def __init__(self, alpha=1.0, solver="auto", random_state=Config.SEED):
        """
        Args:
            alpha (float): Regularization strength.
            solver (str): Solver to use in the computational routines.
            random_state (int): Seed for reproducibility.
        """
        self.model = Ridge(
            alpha=alpha,
            solver=solver,
            random_state=random_state,
            fit_intercept=True,
            copy_X=False,  # Optimization: overwrite X if possible to save memory
        )
        self.path = Config.RIDGE_MODEL_PATH

    def fit(self, X, y):
        """
        Fits the Ridge model.

        Args:
            X (scipy.sparse.csr_matrix): Sparse feature matrix.
            y (numpy.ndarray): Target ranks.
        """
        print(f"Training Ridge Regression on shape {X.shape}...")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts ranks using the fitted model.

        Args:
            X (scipy.sparse.csr_matrix): Sparse feature matrix.

        Returns:
            numpy.ndarray: Predicted ranks.
        """
        return self.model.predict(X)

    def save(self, path=None):
        """Saves the sklearn model to disk using joblib."""
        if path is None:
            path = self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Ridge model saved to {path}")

    def load(self, path=None):
        """Loads the sklearn model from disk."""
        if path is None:
            path = self.path

        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Ridge model loaded from {path}")
        else:
            raise FileNotFoundError(
                f"Ridge model not found at {path}. Call fit() first."
            )
        return self
