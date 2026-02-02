import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops entire channels across the sequence length.
    Input shape: (Batch, SeqLen, Hidden)
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        if not self.training or self.drop_prob == 0:
            return inputs

        # Permute to (Batch, Hidden, SeqLen) for Dropout2d which acts on the channel dim (Hidden)
        output = inputs.permute(0, 2, 1)
        output = F.dropout2d(output, self.drop_prob, training=self.training)
        # Permute back to (Batch, SeqLen, Hidden)
        return output.permute(0, 2, 1)


class AttentionPooling(nn.Module):
    """
    Computes a weighted average of the hidden states based on a learned attention score.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.Tanh(), nn.Linear(in_dim, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (Batch, SeqLen, Hidden)
        # attention_mask: (Batch, SeqLen) - 1 for token, 0 for pad

        # Calculate raw attention scores
        # (Batch, SeqLen, Hidden) -> (Batch, SeqLen, 1) -> (Batch, SeqLen)
        w = self.attention(last_hidden_state).squeeze(-1)

        # Mask padding tokens by setting their score to a very large negative number
        if attention_mask is not None:
            padding_mask = attention_mask == 0
            w.masked_fill_(padding_mask, -1e4)

        # Normalize scores to probabilities
        weights = torch.softmax(w, dim=1)  # (Batch, SeqLen)

        # Weighted sum of hidden states
        # weights.unsqueeze(-1): (Batch, SeqLen, 1)
        # Broadcasting weights across the hidden dimension
        context = torch.sum(
            weights.unsqueeze(-1) * last_hidden_state, dim=1
        )  # (Batch, Hidden)

        return context


class ToxicityModel(nn.Module):
    """
    Multi-Task RoBERTa-Large model with Attention Pooling and Multi-Sample Dropout.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.model_name = Config.MODEL_NAME
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_identity_classes = len(Config.IDENTITY_COLUMNS)
        self.msd_samples = 5  # Number of dropout samples

        # 1. Backbone
        config = AutoConfig.from_pretrained(self.model_name)
        config.output_hidden_states = True
        self.roberta = AutoModel.from_pretrained(self.model_name, config=config)

        # 2. Regularization (Spatial Dropout)
        self.spatial_dropout = SpatialDropout(Config.SPATIAL_DROPOUT)

        # 3. Aggregation (Attention Pooling)
        self.pooling = AttentionPooling(self.hidden_size)

        # 4. Heads
        # Standard Dropout layer used for Multi-Sample Dropout
        self.dropout = nn.Dropout(Config.DROPOUT)

        # Primary Task: Toxicity Classification (Binary)
        self.toxicity_linear = nn.Linear(self.hidden_size, 1)

        # Auxiliary Task: Identity Classification (Multi-label)
        self.identity_linear = nn.Linear(self.hidden_size, self.num_identity_classes)

        # Weight Initialization for new heads
        self._init_weights(self.toxicity_linear)
        self._init_weights(self.identity_linear)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # 1. Extract features from Backbone
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (Batch, SeqLen, Hidden)

        # 2. Apply Spatial Dropout
        embeddings = self.spatial_dropout(last_hidden_state)

        # 3. Apply Attention Pooling
        pooled_output = self.pooling(embeddings, attention_mask)  # (Batch, Hidden)

        # 4. Toxicity Head with Multi-Sample Dropout
        if self.training:
            # Multi-Sample Dropout: Average logits over multiple dropout masks
            toxicity_logits_list = []
            for _ in range(self.msd_samples):
                dropped_output = self.dropout(pooled_output)
                toxicity_logits_list.append(self.toxicity_linear(dropped_output))

            # Stack and mean: (Samples, Batch, 1) -> (Batch, 1)
            toxicity_logits = torch.mean(
                torch.stack(toxicity_logits_list, dim=0), dim=0
            )
        else:
            # Inference: Single pass (weights are already scaled by dropout during training implicitly)
            toxicity_logits = self.toxicity_linear(pooled_output)

        # 5. Identity Head (Auxiliary)
        # Single dropout pass is sufficient for the auxiliary task
        identity_logits = self.identity_linear(self.dropout(pooled_output))

        return {
            "logits": toxicity_logits.squeeze(-1),  # Shape: (Batch,)
            "aux_logits": identity_logits,  # Shape: (Batch, NumIdentities)
        }
