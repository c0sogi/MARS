import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from library.utils import set_seed


class TransformerClassifier(nn.Module):
    """
    A PyTorch module wrapping a pre-trained Transformer for text classification.
    Extracts the CLS token embedding and passes it through a linear layer.
    """

    def __init__(
        self, model_name, num_classes, dropout_rate=0.1, freeze_backbone=False
    ):
        """
        Args:
            model_name (str): Hugging Face model identifier (e.g., 'roberta-base').
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout probability.
            freeze_backbone (bool): If True, freezes the transformer weights.
        """
        super(TransformerClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use the representation of the [CLS] token (first token)
        # We use last_hidden_state[:, 0, :] instead of pooler_output to be
        # compatible with models that don't have a pooler (like DeBERTa).
        cls_token = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_token)
        logits = self.classifier(x)

        return logits


class ClassicalModelWrapper:
    """
    A wrapper class to standardize the API for Scikit-Learn and XGBoost models.
    """

    def __init__(self, model_type, config):
        """
        Args:
            model_type (str): Type of model ('lr', 'nb', 'xgb').
            config (dict): Configuration dictionary with hyperparameters.
        """
        self.model_type = model_type
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        if model_type == "lr":
            # Logistic Regression for sparse features
            self.model = LogisticRegression(
                C=config.get("lr_C", 1.0),
                solver="saga",
                multi_class="multinomial",
                max_iter=1000,
                random_state=self.seed,
                n_jobs=-1,
            )
        elif model_type == "nb":
            # Multinomial Naive Bayes for sparse features
            self.model = MultinomialNB(alpha=config.get("nb_alpha", 0.01))
        elif model_type == "xgb":
            # XGBoost for dense SVD features
            # Note: XGBoost 3.x uses 'device' parameter instead of 'tree_method' for GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = XGBClassifier(
                n_estimators=config.get("xgb_n_estimators", 1000),
                max_depth=config.get("xgb_max_depth", 6),
                learning_rate=config.get("xgb_lr", 0.05),
                subsample=config.get("xgb_subsample", 0.8),
                colsample_bytree=config.get("xgb_colsample", 0.8),
                objective="multi:softprob",
                num_class=3,
                random_state=self.seed,
                n_jobs=-1,
                device=device,
                verbosity=0,  # Silent
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y):
        """
        Fits the underlying model.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Returns:
            np.ndarray: Probabilities [n_samples, n_classes]
        """
        return self.model.predict_proba(X)
