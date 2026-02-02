import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted sum of the last N hidden states of the encoder.
    This allows the model to leverage information from different levels of abstraction.
    """

    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        # Initialize weights to be equal; they will be updated during training
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(torch.tensor([1] * layer_start, dtype=torch.float))
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (embeddings + layers)
        # We select the last 'layer_start' layers
        all_layer_embedding = all_hidden_states[-self.layer_start :]

        # Stack to shape: [num_layers, batch_size, seq_len, hidden_size]
        all_layer_embedding = torch.stack(all_layer_embedding)

        # Calculate softmax weights: [num_layers, 1, 1, 1]
        weight_factor = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)

        # Compute weighted sum across the layer dimension
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0)
        return weighted_average


class LinearAttentionPooling(nn.Module):
    """
    Computes a weighted average of token representations using a learnable attention vector.
    """

    def __init__(self, hidden_size):
        super(LinearAttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [batch, seq_len, hidden]
        # attention_mask: [batch, seq_len]

        # Calculate attention scores: [batch, seq_len]
        w = self.attention(last_hidden_state).squeeze(-1)

        # Mask padding tokens so they don't contribute to the average
        # Use a large negative number for stability in softmax
        w.masked_fill_(attention_mask == 0, -1e9)

        # Normalize scores to probabilities
        att_weights = F.softmax(w, dim=1)

        # Weighted sum: [batch, hidden]
        # Expand weights to [batch, seq_len, 1] for broadcasting
        context_vector = torch.sum(last_hidden_state * att_weights.unsqueeze(-1), dim=1)

        return context_vector


class CustomDeberta(nn.Module):
    """
    Main model architecture combining DeBERTa-v3, Weighted Layer Pooling,
    Hybrid Pooling (Attention + Max), and Multi-Sample Dropout.
    """

    def __init__(self, pretrained_path=None):
        super(CustomDeberta, self).__init__()

        # Load configuration
        # If pretrained_path is provided (e.g., from MLM), use it.
        # Otherwise use the base model name from Config.
        model_source = pretrained_path if pretrained_path else Config.model_name
        self.config = AutoConfig.from_pretrained(model_source)

        # Update config with specific dropout and label settings
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": Config.hidden_dropout_prob,
                "attention_probs_dropout_prob": Config.attention_probs_dropout_prob,
                "num_labels": Config.num_classes,
            }
        )

        # Load Backbone
        self.model = AutoModel.from_pretrained(model_source, config=self.config)

        # 1. Weighted Layer Pooling
        self.layer_pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers,
            layer_start=Config.num_layers_to_aggregate,
        )

        # 2. Hybrid Pooling Components
        self.attention_pooler = LinearAttentionPooling(self.config.hidden_size)

        # 3. Multi-Sample Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.hidden_dropout_prob) for _ in range(Config.msd_num)]
        )

        # 4. Classification Head
        # Input dimension is hidden_size * 2 because we concatenate Attention Pool and Max Pool
        self.fc = nn.Linear(self.config.hidden_size * 2, Config.num_classes)

        # Initialize custom layers
        self._init_weights(self.fc)
        self._init_weights(self.attention_pooler.attention)

    def _init_weights(self, module):
        """
        Standard weight initialization for Transformer-based models.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # 1. Weighted Layer Aggregation
        # Result shape: [batch, seq_len, hidden]
        sequence_output = self.layer_pooler(all_hidden_states)

        # 2. Hybrid Pooling
        # A. Linear Attention Pooling
        att_pool = self.attention_pooler(sequence_output, attention_mask)

        # B. Global Max Pooling
        # We must mask the padding tokens before max pooling to avoid selecting padding artifacts
        # Create mask: [batch, seq_len, 1]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        )

        # Clone to avoid modifying gradients in place improperly
        sequence_output_masked = sequence_output.clone()
        # Set padding positions to a very small number
        sequence_output_masked[input_mask_expanded == 0] = -1e9

        # Max over sequence dimension
        max_pool = torch.max(sequence_output_masked, 1)[0]

        # Concatenate pooling outputs
        logits_input = torch.cat([att_pool, max_pool], dim=1)

        # 3. Multi-Sample Dropout & Classification
        # Pass through each dropout mask, classify, and stack results
        # Shape after stack: [msd_num, batch, num_classes]
        multi_logits = torch.stack(
            [self.fc(dp(logits_input)) for dp in self.dropouts], dim=0
        )

        # Average the logits
        logits = torch.mean(multi_logits, dim=0)

        # 4. Loss Calculation
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
