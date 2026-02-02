import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import Config


class ChatbotBiEncoder(nn.Module):
    """
    Bi-Encoder (Siamese) Architecture for Chatbot Preference Prediction.

    Encodes Prompt, Response A, and Response B independently using a shared Transformer backbone.
    Constructs a feature vector from the pooled embeddings and their interactions,
    then passes it through a classification head.

    Cite solution_lesson_node_00001: Fine-tuning the encoder instead of using frozen embeddings.
    Cite solution_lesson_node_00002: Using Bi-Encoder to avoid truncation issues with long sequences.
    """

    def __init__(
        self,
        model_name: str = Config.TRANSFORMER_MODEL,
        hidden_layers: list = None,
        dropout_rate: float = None,
        num_classes: int = None,
    ):
        super(ChatbotBiEncoder, self).__init__()

        # Load Transformer Backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        embedding_dim = self.backbone.config.hidden_size

        # Set defaults
        if hidden_layers is None:
            hidden_layers = Config.HIDDEN_LAYERS
        if dropout_rate is None:
            dropout_rate = Config.DROPOUT_RATE
        if num_classes is None:
            num_classes = Config.NUM_CLASSES

        # Input dim: Prompt + ResA + ResB + Diff + Prod
        input_dim = embedding_dim * 5

        # Classification Head
        layers = []
        current_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, features: dict) -> torch.Tensor:
        """
        Forward pass.
        Args:
            features (dict): Dictionary containing input_ids and attention_mask for 'prompt', 'res_a', 'res_b'.
        """
        # Encode Prompt
        p_out = self.backbone(
            input_ids=features["prompt_input_ids"],
            attention_mask=features["prompt_attention_mask"],
        )
        p_emb = self.mean_pooling(p_out, features["prompt_attention_mask"])

        # Encode Response A
        a_out = self.backbone(
            input_ids=features["res_a_input_ids"],
            attention_mask=features["res_a_attention_mask"],
        )
        a_emb = self.mean_pooling(a_out, features["res_a_attention_mask"])

        # Encode Response B
        b_out = self.backbone(
            input_ids=features["res_b_input_ids"],
            attention_mask=features["res_b_attention_mask"],
        )
        b_emb = self.mean_pooling(b_out, features["res_b_attention_mask"])

        # Interactions
        diff = torch.abs(a_emb - b_emb)
        prod = a_emb * b_emb

        # Concatenate
        combined = torch.cat([p_emb, a_emb, b_emb, diff, prod], dim=1)

        # Classify
        return self.head(combined)
