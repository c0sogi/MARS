import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class NBSVM(BaseEstimator, ClassifierMixin):
    """
    NBSVM (Naive Bayes - Support Vector Machine) variant using Logistic Regression.
    This implementation fits a separate binary classifier for each label (Binary Relevance)
    using Naive Bayes log-count ratios as feature scaling.
    """

    def __init__(self, C=1.0, dual=False, n_jobs=1, random_state=42, max_iter=1000):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_iter = max_iter
        self.models = []
        self.r_values = []

    def _calculate_r(self, X, y):
        """
        Computes the Naive Bayes log-count ratio 'r'.
        r = log( (p/|p|) / (q/|q|) )
        where p is the feature count vector for positive class (smoothed)
        and q is the feature count vector for negative class (smoothed).
        """
        # Ensure X is CSR for efficient slicing
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()

        # Sum features for positive and negative classes
        # Adding 1 for Laplace smoothing
        p = X[y == 1].sum(axis=0) + 1
        q = X[y == 0].sum(axis=0) + 1

        # Normalize to probabilities
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate log ratio
        r = np.log(p / q)
        return r

    def fit(self, X, y):
        """
        Fits the NBSVM model.

        Args:
            X: Sparse matrix of shape (n_samples, n_features)
            y: Binary matrix of shape (n_samples, n_classes) or array (n_samples,)
        """
        self.models = []
        self.r_values = []

        # Ensure X is sparse
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        # Handle single target case by reshaping
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_classes = y.shape[1]

        for i in range(n_classes):
            y_i = y[:, i]

            # Compute r for this class
            r = self._calculate_r(X, y_i)
            self.r_values.append(r)

            # Scale X element-wise by r
            # X is (N, V), r is (1, V). multiply broadcasts correctly.
            X_nb = X.multiply(r)

            # Fit Logistic Regression
            # 'liblinear' is required for dual=True, 'lbfgs' is standard for dual=False
            solver = "liblinear" if self.dual else "lbfgs"

            clf = LogisticRegression(
                C=self.C,
                dual=self.dual,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                solver=solver,
                max_iter=self.max_iter,
            )
            clf.fit(X_nb, y_i)
            self.models.append(clf)

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities for each class.

        Args:
            X: Sparse matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ["models", "r_values"])

        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        preds = []
        for i, model in enumerate(self.models):
            r = self.r_values[i]

            # Scale X with the stored r for this class
            X_nb = X.multiply(r)

            # Get probability of positive class (index 1)
            prob = model.predict_proba(X_nb)[:, 1]
            preds.append(prob)

        # Stack predictions column-wise
        return np.column_stack(preds)


class ToxicRoBERTa(nn.Module):
    """
    RoBERTa-based model for multi-label toxicity classification.
    Wraps a pre-trained HuggingFace RoBERTa model.
    """

    def __init__(self, model_name="roberta-base", num_classes=6, dropout_rate=0.1):
        super(ToxicRoBERTa, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, ids, mask):
        """
        Forward pass of the model.

        Args:
            ids (torch.Tensor): Input IDs tensor of shape (batch_size, seq_len)
            mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Get RoBERTa outputs
        outputs = self.roberta(input_ids=ids, attention_mask=mask)

        # Extract the [CLS] token representation (first token in the sequence)
        # last_hidden_state has shape (batch_size, seq_len, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout for regularization
        x = self.dropout(cls_output)

        # Pass through linear classification head
        logits = self.classifier(x)

        return logits
