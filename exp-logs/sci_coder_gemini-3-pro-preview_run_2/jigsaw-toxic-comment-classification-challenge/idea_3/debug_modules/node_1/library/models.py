import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout.
    Applies multiple dropout masks with different rates to the input,
    passes each through a shared linear layer, and averages the outputs.
    """

    def __init__(self, in_features, out_features, dropout_rates):
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.linear(dropout(x))
            else:
                out += self.linear(dropout(x))
        return out / len(self.dropouts)


class CustomTransformer(nn.Module):
    """
    Transformer-based model for Toxicity Prediction.
    Uses a pre-trained backbone (e.g., DeBERTa-v3, RoBERTa) and a
    Multi-Sample Dropout classification head.
    """

    def __init__(self, model_name, config=Config):
        super(CustomTransformer, self).__init__()
        self.config = config

        # Load AutoConfig to get hidden_size
        self.model_config = AutoConfig.from_pretrained(model_name)

        # Load the backbone
        self.model = AutoModel.from_pretrained(model_name, config=self.model_config)

        # Classification Head
        # Using the hidden size from the config (usually 768 for base models)
        self.head = MultiSampleDropout(
            in_features=self.model_config.hidden_size,
            out_features=self.config.num_classes,
            dropout_rates=self.config.msd_rates,
        )

        # Initialize weights of the head
        self._init_weights(self.head.linear)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, ids, mask, token_type_ids=None):
        # Forward pass through the backbone
        # Some models (like RoBERTa) don't use token_type_ids, but passing them
        # usually doesn't hurt if the model handles arguments flexibly.
        # However, to be safe and precise:
        kwargs = {
            "input_ids": ids,
            "attention_mask": mask,
        }

        # DeBERTa and BERT use token_type_ids; RoBERTa does not.
        # We check the model type or just pass it if provided, relying on transformers to ignore or use.
        # DeBERTa V3 expects token_type_ids usually.
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.model(**kwargs)

        # Extract the representation of the [CLS] token (first token)
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Pass through the Multi-Sample Dropout head
        logits = self.head(cls_embedding)

        return logits


class LinearModelWrapper:
    """
    Wrapper for the Linear Baseline Model (Logistic Regression).
    Uses OneVsRestClassifier to handle multi-label classification.
    """

    def __init__(self):
        # C=4.0 is a strong baseline hyperparameter for this specific dataset based on literature
        # solver='liblinear' is efficient for high-dimensional sparse data
        self.model = OneVsRestClassifier(
            LogisticRegression(
                C=4.0, solver="liblinear", random_state=Config.seed, max_iter=1000
            )
        )

    def fit(self, X, y):
        """
        Fits the linear model.

        Args:
            X: Sparse matrix of features.
            y: Binary labels (N, num_classes).
        """
        print("Fitting Linear Model (Logistic Regression)...")
        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Predicts probabilities.

        Args:
            X: Sparse matrix of features.

        Returns:
            numpy array of shape (N, num_classes) with probabilities.
        """
        return self.model.predict_proba(X)
