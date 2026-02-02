import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from library.config import Config


class CustomTransformer(nn.Module):
    """
    A PyTorch module wrapping a Hugging Face Transformer backbone with a custom
    Multi-Layer Concatenation head.

    This architecture extracts the [CLS] tokens from the last 4 hidden layers
    of the transformer, concatenates them, and passes them through a classification
    head. This preserves hierarchical features (morphology, syntax, semantics)
    often lost when using only the final layer.

    Args:
        model_name (str): The Hugging Face model identifier (e.g., 'microsoft/deberta-v3-base').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained weights.
        dropout_p (float): Dropout probability for the classification head.
    """

    def __init__(self, model_name, num_classes=3, pretrained=True, dropout_p=0.1):
        super(CustomTransformer, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.use_multi_layer_concat = Config.USE_MULTI_LAYER_CONCAT

        # Load Configuration
        # output_hidden_states=True is required to access intermediate layers
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True

        # Adjust dropout in config if needed, though we apply custom dropout in head
        config.hidden_dropout_prob = dropout_p
        config.attention_probs_dropout_prob = dropout_p

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=config)
        else:
            self.backbone = AutoModel.from_config(config)

        self.hidden_size = config.hidden_size

        # Define Head Architecture
        if self.use_multi_layer_concat:
            # Concatenating [CLS] from the last 4 layers
            self.concat_hidden_size = self.hidden_size * 4
        else:
            # Standard approach: Only the last layer
            self.concat_hidden_size = self.hidden_size

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(self.concat_hidden_size, num_classes)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """Initialize the weights of the classification head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits for the classes.
        """
        # Pass inputs through the transformer backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_multi_layer_concat:
            # outputs.hidden_states is a tuple of tensors for all layers
            # We access the last 4 layers using negative indexing
            hidden_states = outputs.hidden_states

            # Extract [CLS] token (index 0) from the last 4 layers
            # Note: DeBERTa, RoBERTa, and BERT all use index 0 for the [CLS]/start token
            cls_embeddings = []
            # Iterate from the last layer backwards to the 4th last
            for i in range(1, 5):  # Indices: -1, -2, -3, -4
                layer_output = hidden_states[-i]
                # layer_output shape: (batch_size, seq_len, hidden_size)
                cls_token = layer_output[:, 0, :]  # Shape: (batch_size, hidden_size)
                cls_embeddings.append(cls_token)

            # Concatenate the extracted embeddings along the feature dimension
            # Shape: (batch_size, hidden_size * 4)
            feature = torch.cat(cls_embeddings, dim=1)
        else:
            # Standard approach: Use the [CLS] token from the last hidden state
            last_hidden_state = outputs.last_hidden_state
            feature = last_hidden_state[:, 0, :]

        # Classification Head
        x = self.dropout(feature)
        logits = self.fc(x)

        return logits


class StatisticalModel(BaseEstimator, ClassifierMixin):
    """
    A weighted ensemble of Logistic Regression and Multinomial Naive Bayes.

    This model is designed to work with sparse TF-IDF features. It combines
    the discriminative power of Logistic Regression with the generative
    baseline of Naive Bayes, which is often effective for text classification.

    Args:
        lr_C (float): Inverse regularization strength for Logistic Regression.
        nb_alpha (float): Additive smoothing parameter for Naive Bayes.
        weight_lr (float): Weight assigned to Logistic Regression probabilities (0.0 to 1.0).
                           Naive Bayes gets (1.0 - weight_lr).
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, lr_C=1.0, nb_alpha=1.0, weight_lr=0.5, random_state=42):
        self.lr_C = lr_C
        self.nb_alpha = nb_alpha
        self.weight_lr = weight_lr
        self.random_state = random_state

        # Initialize sub-models
        # solver='liblinear' is chosen for its efficiency with high-dimensional sparse data
        self.lr_model = LogisticRegression(
            C=self.lr_C,
            solver="liblinear",
            multi_class="ovr",
            random_state=self.random_state,
        )
        self.nb_model = MultinomialNB(alpha=self.nb_alpha)

        self.classes_ = None

    def fit(self, X, y):
        """
        Fit both Logistic Regression and Naive Bayes models on the training data.

        Args:
            X (scipy.sparse.csr_matrix): Training features.
            y (numpy.ndarray): Target labels.

        Returns:
            self
        """
        self.lr_model.fit(X, y)
        self.nb_model.fit(X, y)
        self.classes_ = self.lr_model.classes_
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (scipy.sparse.csr_matrix): Input features.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X using the weighted ensemble.

        Args:
            X (scipy.sparse.csr_matrix): Input features.

        Returns:
            numpy.ndarray: Weighted class probabilities.
        """
        # Get probabilities from individual models
        p_lr = self.lr_model.predict_proba(X)
        p_nb = self.nb_model.predict_proba(X)

        # Calculate weighted average
        # Ensure weight_lr is clamped between 0 and 1 for safety
        w = max(0.0, min(1.0, self.weight_lr))
        p_final = w * p_lr + (1.0 - w) * p_nb

        return p_final
