import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import MODEL_NAME, TARGET_COLS


class SiameseNetwork(nn.Module):
    """
    Siamese Network with a Transformer backbone (MPNet).
    Encodes Question and Answer separately, computes interaction features,
    and predicts target probabilities via an MLP head.
    """

    def __init__(
        self, model_name=MODEL_NAME, num_labels=len(TARGET_COLS), dropout_rate=0.1
    ):
        super(SiameseNetwork, self).__init__()

        # Load Transformer Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Dimension of the embedding output from the backbone
        self.hidden_size = self.config.hidden_size

        # Interaction features dimension:
        # [u, v, |u-v|, u*v] -> 4 vectors concatenated
        self.interaction_dim = self.hidden_size * 4

        # MLP Head for classification/regression
        self.classifier = nn.Sequential(
            nn.Linear(self.interaction_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, num_labels),
        )

    def mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling on the token embeddings, accounting for the attention mask.
        Standard approach for Sentence-Transformers.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embedding(self, input_ids, attention_mask):
        """
        Passes input through backbone and performs pooling to get sentence embedding.
        """
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pooling(output, attention_mask)

    def _compute_interaction_features(self, u, v):
        """
        Computes the interaction vector X = [u, v, |u-v|, u*v].
        """
        diff_uv = torch.abs(u - v)
        prod_uv = u * v
        return torch.cat([u, v, diff_uv, prod_uv], dim=1)

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass for end-to-end training.
        Returns logits (unnormalized scores).
        """
        # Encode Question (u) and Answer (v)
        u = self.get_embedding(q_input_ids, q_attention_mask)
        v = self.get_embedding(a_input_ids, a_attention_mask)

        # Compute Interaction Features
        features = self._compute_interaction_features(u, v)

        # Predict
        logits = self.classifier(features)
        return logits

    def extract_features(
        self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
    ):
        """
        Extracts the interaction features without passing them through the classifier.
        Used for generating cached features for Stage 2 (Ridge Regression).
        """
        # Ensure gradients are not computed for feature extraction
        with torch.no_grad():
            u = self.get_embedding(q_input_ids, q_attention_mask)
            v = self.get_embedding(a_input_ids, a_attention_mask)
            features = self._compute_interaction_features(u, v)

        return features
