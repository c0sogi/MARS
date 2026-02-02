import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from library.config import Config


class RandomForestModel:
    """
    Wrapper for the Random Forest component of the ensemble.
    Uses hyperparameters defined in Config.RF_PARAMS.
    """

    def __init__(self):
        self.clf = RandomForestClassifier(**Config.RF_PARAMS)

    def fit(self, X, y):
        """
        Fits the Random Forest model.
        Args:
            X (sparse matrix or array): Feature matrix.
            y (array): Target labels.
        """
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """
        Predicts class probabilities.
        Args:
            X (sparse matrix or array): Feature matrix.
        Returns:
            array: Probabilities for the positive class (1).
        """
        # predict_proba returns [prob_0, prob_1], we want prob_1
        return self.clf.predict_proba(X)[:, 1]


class FiLMClassifier(nn.Module):
    """
    Neural Network with FiLM (Feature-wise Linear Modulation) Fusion.

    Architecture:
    1. Semantic Branches: Title, Body, History (Dual Attention), Centroid.
    2. Control Branch: Metadata + Sentiment + TopK Subreddits.
    3. FiLM Mechanism: Control branch modulates the semantic representation.
    """

    def __init__(self, control_input_dim):
        """
        Args:
            control_input_dim (int): Dimension of the control feature vector
                                     (Meta + TopK + VADER).
        """
        super(FiLMClassifier, self).__init__()

        self.text_dim = Config.TEXT_EMBED_DIM
        self.film_hidden_dim = Config.FILM_HIDDEN_DIM

        # ---------------------------------------------------------------------
        # 1. Attention Mechanism (Dual-Query)
        # ---------------------------------------------------------------------
        # We compute attention manually using dot products, so no learnable
        # parameters needed here unless we want to project queries/keys.
        # Given the "Raw SBERT" instruction, we stick to dot-product attention
        # without projection layers for the embeddings themselves initially.

        # ---------------------------------------------------------------------
        # 2. Main Semantic Feature Assembly
        # ---------------------------------------------------------------------
        # Components:
        # 1. Title Emb (384)
        # 2. Body Emb (384)
        # 3. History Context via Title Query (384)
        # 4. History Context via Body Query (384)
        # 5. Global Centroid (384)
        self.semantic_feature_dim = self.text_dim * 5

        # ---------------------------------------------------------------------
        # 3. FiLM Generator (Control Branch)
        # ---------------------------------------------------------------------
        # Maps control_features -> gamma (scale) and beta (shift) for semantic features
        self.film_generator = nn.Sequential(
            nn.Linear(control_input_dim, self.film_hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            # Output dim is 2 * semantic_feature_dim (one for gamma, one for beta)
            nn.Linear(self.film_hidden_dim, 2 * self.semantic_feature_dim),
        )

        # ---------------------------------------------------------------------
        # 4. Classifier Head
        # ---------------------------------------------------------------------
        # Input: Modulated Semantic Features + Original Control Features (Skip Connection)
        head_input_dim = self.semantic_feature_dim + control_input_dim

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(64, 1),
        )

        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(Config.MLP_DROPOUT_EMBED)

    def _attention(self, query, keys, mask):
        """
        Computes weighted sum of keys based on dot-product similarity with query.

        Args:
            query: (B, D)
            keys: (B, SeqLen, D)
            mask: (B, SeqLen) - 1 for valid, 0 for pad

        Returns:
            context: (B, D)
        """
        # Expand query to (B, D, 1) for batch matrix multiplication
        # keys is (B, SeqLen, D)
        # Scores: (B, SeqLen, 1) = bmm(keys, query.unsqueeze(2))
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # (B, SeqLen)

        # Apply mask: Set pad positions to -inf
        # mask is 1.0 for valid, 0.0 for pad
        # We want to replace 0.0 with -1e9
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=1)  # (B, SeqLen)

        # Weighted sum
        # weights: (B, 1, SeqLen)
        # keys: (B, SeqLen, D)
        # context: (B, 1, D) -> squeeze -> (B, D)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)

        return context

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Dictionary containing:
                - 'title_emb': (B, 384)
                - 'body_emb': (B, 384)
                - 'history_emb': (B, SeqLen, 384)
                - 'history_mask': (B, SeqLen)
                - 'global_centroid': (B, 384)
                - 'control_features': (B, ControlDim)
        """
        # Unpack
        title = inputs["title_emb"]
        body = inputs["body_emb"]
        history = inputs["history_emb"]
        mask = inputs["history_mask"]
        centroid = inputs["global_centroid"]
        control = inputs["control_features"]

        # Apply dropout to embeddings
        title = self.embed_dropout(title)
        body = self.embed_dropout(body)
        history = self.embed_dropout(history)
        centroid = self.embed_dropout(centroid)

        # 1. Dual-Query Attention
        # Context 1: Title queries History
        context_title = self._attention(title, history, mask)

        # Context 2: Body queries History
        context_body = self._attention(body, history, mask)

        # 2. Concatenate Semantic Features
        # [Title, Body, Context_Title, Context_Body, Centroid]
        semantic_features = torch.cat(
            [title, body, context_title, context_body, centroid], dim=1
        )

        # 3. Generate FiLM Parameters
        film_params = self.film_generator(control)

        # Split into gamma (scale) and beta (shift)
        gamma, beta = torch.chunk(film_params, 2, dim=1)

        # 4. Apply FiLM Modulation
        # Formula: (1 + gamma) * features + beta
        modulated_features = (1 + gamma) * semantic_features + beta

        # 5. Skip Connection & Classifier Head
        # Concatenate modulated features with the raw control features
        combined = torch.cat([modulated_features, control], dim=1)

        logits = self.head(combined)

        return logits
