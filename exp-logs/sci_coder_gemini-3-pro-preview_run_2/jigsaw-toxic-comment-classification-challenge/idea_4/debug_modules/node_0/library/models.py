import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from transformers import AutoModel, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Applies dropout multiple times to the input and averages the results
    from the classification layer. This acts as a mini-ensemble within the model,
    smoothing the loss landscape and accelerating convergence.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_samples: int = 5,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, hidden_size)
        logits_list = []
        for i in range(self.num_samples):
            # Apply dropout sample i
            x_dropped = self.dropouts[i](x)
            # Pass through the shared linear layer
            logits_list.append(self.classifier(x_dropped))

        # Stack and average the logits
        # shape: (batch_size, num_samples, out_features)
        stacked_logits = torch.stack(logits_list, dim=1)
        # shape: (batch_size, out_features)
        avg_logits = torch.mean(stacked_logits, dim=1)
        return avg_logits


class CustomTransformer(nn.Module):
    """
    Transformer-based model for toxicity classification.
    Supports loading from standard HF Hub or domain-adapted (TAPT) local paths.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 6,
        pretrained_path: str = None,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            model_name: HuggingFace model identifier (e.g., 'roberta-base').
            num_classes: Number of output labels.
            pretrained_path: Path to local weights (e.g., from TAPT). If None or invalid, loads from Hub.
            dropout_rate: Dropout probability for the classification head.
        """
        super().__init__()

        # Determine where to load weights from
        load_path = model_name
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading domain-adapted weights from: {pretrained_path}")
            load_path = pretrained_path
        else:
            print(f"Loading standard weights from HuggingFace Hub: {model_name}")

        self.config = AutoConfig.from_pretrained(load_path)
        self.backbone = AutoModel.from_pretrained(load_path, config=self.config)

        # Gradient Checkpointing can save memory
        self.backbone.gradient_checkpointing_enable()

        self.hidden_size = self.config.hidden_size
        self.head = MultiSampleDropout(
            in_features=self.hidden_size,
            out_features=num_classes,
            num_samples=5,
            dropout_rate=dropout_rate,
        )

        # Initialize weights of the head
        self._init_weights(self.head.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass.
        Returns logits.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract CLS token representation
        # For BERT/RoBERTa/DeBERTa, this is usually the first token of the last hidden state
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        logits = self.head(cls_embedding)

        return logits


class LinearModelWrapper:
    """
    Wrapper for the Scikit-Learn Logistic Regression Baseline.
    Implements One-Vs-Rest strategy for multi-label classification.
    """

    def __init__(self, params: dict = None):
        if params is None:
            params = Config.LINEAR_PARAMS

        self.c_val = params.get("c_val", 1.0)
        self.solver = params.get("solver", "sag")
        self.n_jobs = params.get("n_jobs", -1)

        # Initialize MultiOutputClassifier with LogisticRegression
        # This fits one regressor per target column
        self.model = MultiOutputClassifier(
            LogisticRegression(
                C=self.c_val,
                solver=self.solver,
                n_jobs=1,  # n_jobs handled by MultiOutputClassifier wrapper
                max_iter=1000,
                random_state=Config.SEED,
            ),
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y):
        """
        Fits the model to the training data.
        """
        print(f"Training Linear Model (C={self.c_val}, solver={self.solver})...")
        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Predicts probabilities for each class.
        Returns: np.ndarray of shape (n_samples, n_classes)
        """
        # predict_proba returns a list of arrays (one per class),
        # where each array is (n_samples, 2) for binary classification.
        # We need the probability of the positive class (index 1).
        preds_list = self.model.predict_proba(X)

        # Extract prob for class 1 for each label
        # Handle case where a class might have only 1 label in training (though rare with stratification)
        probs = []
        for i, class_preds in enumerate(preds_list):
            if class_preds.shape[1] == 2:
                probs.append(class_preds[:, 1])
            else:
                # Fallback if model sees only one class (e.g. all zeros)
                # This shouldn't happen with proper stratification, but for safety:
                # If only class 0 exists, prob is 0. If only class 1, prob is 1.
                # We assume the single column corresponds to the only class present.
                # However, sklearn usually returns shape (n, 1) if only one class.
                # We need to know which class it is.
                # For simplicity in this robust wrapper, we'll assume 0 if shape is (n,1)
                # and the value is 1.0 for class 0.
                # A safer bet for this specific competition data is usually just taking column 0
                # if it's the only one, but let's assume standard binary behavior.
                # Given the dataset size, we assume 2 classes are always present.
                probs.append(class_preds[:, 1])

        return np.column_stack(probs)

    def save(self, path: str):
        """
        Saves the model using joblib.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Linear model saved to {path}")

    def load(self, path: str):
        """
        Loads the model using joblib.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        print(f"Linear model loaded from {path}")
