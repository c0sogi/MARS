import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import Config


class SiameseBiEncoder(nn.Module):
    """
    A Siamese Bi-Encoder model that encodes anchor and target phrases independently
    using a shared Transformer backbone, computes their cosine similarity, and
    scales the result using a linear layer.
    """

    def __init__(self, model_name=None):
        """
        Args:
            model_name (str, optional): Name of the pre-trained model to load.
                                        Defaults to Config.model_name.
        """
        super(SiameseBiEncoder, self).__init__()

        if model_name is None:
            model_name = Config.model_name

        # Load the pre-trained Transformer backbone
        self.backbone = AutoModel.from_pretrained(model_name)

        # Learnable linear scaling: score = w * cos(u, v) + b
        # Input dim: 1 (cosine similarity), Output dim: 1 (score)
        self.linear = nn.Linear(1, 1)

        # Initialize weights to map cosine range [-1, 1] to score range [0, 1]
        # Formula: 0.5 * cos + 0.5
        # If cos = 1, score = 1.0
        # If cos = -1, score = 0.0
        nn.init.constant_(self.linear.weight, 0.5)
        nn.init.constant_(self.linear.bias, 0.5)

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Performs mean pooling on token embeddings, accounting for the attention mask.

        Args:
            token_embeddings (torch.Tensor): Output from transformer [Batch, SeqLen, Hidden].
            attention_mask (torch.Tensor): Attention mask [Batch, SeqLen].

        Returns:
            torch.Tensor: Pooled embeddings [Batch, Hidden].
        """
        # Expand mask to match embedding dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Sum embeddings and mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Divide by valid token count
        return sum_embeddings / sum_mask

    def forward(
        self,
        anchor_input_ids,
        anchor_attention_mask,
        target_input_ids,
        target_attention_mask,
    ):
        """
        Forward pass for the Siamese network.

        Args:
            anchor_input_ids (torch.Tensor): Input IDs for anchor phrases.
            anchor_attention_mask (torch.Tensor): Attention masks for anchor phrases.
            target_input_ids (torch.Tensor): Input IDs for target phrases.
            target_attention_mask (torch.Tensor): Attention masks for target phrases.

        Returns:
            torch.Tensor: Predicted similarity scores [Batch].
        """
        # 1. Encode Anchor
        anchor_out = self.backbone(
            input_ids=anchor_input_ids, attention_mask=anchor_attention_mask
        )
        anchor_emb = self.mean_pooling(
            anchor_out.last_hidden_state, anchor_attention_mask
        )

        # 2. Encode Target
        target_out = self.backbone(
            input_ids=target_input_ids, attention_mask=target_attention_mask
        )
        target_emb = self.mean_pooling(
            target_out.last_hidden_state, target_attention_mask
        )

        # 3. Compute Cosine Similarity
        # Output shape: [Batch]
        cosine_sim = torch.cosine_similarity(anchor_emb, target_emb)

        # 4. Linear Scaling
        # Reshape to [Batch, 1] for Linear layer, then squeeze back to [Batch]
        score = self.linear(cosine_sim.unsqueeze(1)).squeeze(1)

        return score
