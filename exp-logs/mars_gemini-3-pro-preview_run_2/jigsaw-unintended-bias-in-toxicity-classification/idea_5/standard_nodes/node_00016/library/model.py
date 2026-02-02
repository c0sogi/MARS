import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops entire channels across the sequence length.
    Standard nn.Dropout drops individual elements.
    Input shape: (Batch, Seq_Len, Hidden_Dim)
    """

    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        if not self.training or self.drop_prob == 0:
            return inputs

        # Inputs: (B, L, D)
        # Permute to (B, D, L, 1) to use Dropout2d which expects (N, C, H, W)
        inputs = inputs.permute(0, 2, 1).unsqueeze(3)

        # Apply dropout to the 'C' dimension (Hidden_Dim)
        inputs = nn.functional.dropout2d(
            inputs, p=self.drop_prob, training=self.training
        )

        # Restore shape: (B, D, L, 1) -> (B, D, L) -> (B, L, D)
        inputs = inputs.squeeze(3).permute(0, 2, 1)
        return inputs


class AttentionPooling(nn.Module):
    """
    Computes a weighted average of the hidden states based on learned attention scores.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (B, L, D)
        # attention_mask: (B, L)

        # Calculate attention scores: (B, L, 1)
        w = self.attention(last_hidden_state)

        # Squeeze to (B, L)
        w = w.squeeze(-1)

        # Mask padding tokens so they don't contribute to the average
        # attention_mask is 1 for tokens, 0 for padding.
        # We set padding positions to a very large negative number for Softmax
        if attention_mask is not None:
            # Create a mask where padding is True
            padding_mask = attention_mask == 0
            w = w.masked_fill(padding_mask, -1e9)

        # Softmax to get probabilities
        weights = torch.softmax(w, dim=1)

        # Weighted sum: (B, L, 1) * (B, L, D) -> (B, L, D) -> sum -> (B, D)
        context_vector = torch.sum(weights.unsqueeze(-1) * last_hidden_state, dim=1)

        return context_vector


class ToxicityModel(nn.Module):
    """
    RoBERTa-Large with Spatial Dropout, Attention Pooling, and Multi-Sample Dropout.
    Outputs both Toxicity logits and Auxiliary Identity logits.
    """

    def __init__(self):
        super(ToxicityModel, self).__init__()

        # Load Configuration and Backbone
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        config.output_hidden_states = True
        self.roberta = AutoModel.from_pretrained(Config.MODEL_NAME, config=config)

        # Feature Extraction Components
        self.spatial_dropout = SpatialDropout(Config.SPATIAL_DROPOUT)
        self.pooling = AttentionPooling(Config.HIDDEN_SIZE)

        # Multi-Sample Dropout Components
        # We create a list of dropout layers to apply different masks
        self.ms_dropouts = nn.ModuleList(
            [nn.Dropout(Config.DROPOUT) for _ in range(Config.MULTI_SAMPLE_DROPOUT_NUM)]
        )

        # Classification Heads
        # Primary Task: Toxicity (1 output)
        self.linear = nn.Linear(Config.HIDDEN_SIZE, Config.NUM_LABELS)

        # Auxiliary Task: Identities (9 outputs)
        self.aux_linear = nn.Linear(Config.HIDDEN_SIZE, Config.NUM_AUX_LABELS)

        # Initialize weights for new layers
        self._init_weights(self.linear)
        self._init_weights(self.aux_linear)
        self._init_weights(self.pooling.attention)

    def _init_weights(self, module):
        """Initialize weights for linear layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)

    def forward(self, input_ids, attention_mask):
        # 1. Backbone Forward Pass
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Get sequence of hidden states: (B, L, D)
        last_hidden_state = outputs.last_hidden_state

        # 2. Spatial Dropout
        # Applied to the sequence before pooling to encourage distributed representations
        last_hidden_state = self.spatial_dropout(last_hidden_state)

        # 3. Attention Pooling
        # Aggregate sequence into a single vector: (B, D)
        pooled_output = self.pooling(last_hidden_state, attention_mask)

        # 4. Multi-Sample Dropout & Prediction
        # Apply multiple dropout masks and average the predictions
        toxicity_logits_list = []
        aux_logits_list = []

        for dropout_layer in self.ms_dropouts:
            # Apply specific dropout mask
            dropped_output = dropout_layer(pooled_output)

            # Pass through shared heads
            tox_logits = self.linear(dropped_output)
            aux_logits = self.aux_linear(dropped_output)

            toxicity_logits_list.append(tox_logits)
            aux_logits_list.append(aux_logits)

        # Stack and Mean
        # Shape: (B, 1)
        final_toxicity_logits = torch.mean(torch.stack(toxicity_logits_list), dim=0)

        # Shape: (B, Num_Aux)
        final_aux_logits = torch.mean(torch.stack(aux_logits_list), dim=0)

        return final_toxicity_logits, final_aux_logits
