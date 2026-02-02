import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import CFG


class WeightedLayerPooling(nn.Module):
    """
    Computes a learned weighted average of the last N hidden layers of the Transformer.
    """

    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers + 1), dtype=torch.float)
            )
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (Batch, SeqLen, Hidden)
        # We take the last 'layer_start' layers
        all_layer_embedding = all_hidden_states[-self.layer_start :]
        all_layer_embedding = torch.stack(
            all_layer_embedding
        )  # (N_layers, Batch, SeqLen, Hidden)

        # Compute softmax weights
        weight_factor = (
            self.layer_weights[-self.layer_start :]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(all_layer_embedding.size())
        )
        weights = torch.nn.functional.softmax(weight_factor, dim=0)

        # Weighted sum
        weighted_average = (weights * all_layer_embedding).sum(
            dim=0
        )  # (Batch, SeqLen, Hidden)
        return weighted_average


class EssayModel(nn.Module):
    """
    Deep Semantic Branch Model.
    Backbone: DeBERTa-v3-Large
    Pooling: Weighted Layer Pooling -> Concatenated Mean & Max Pooling
    Head: Linear Regression
    """

    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                CFG.model_name, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)

        # Ensure hidden states are outputted for WeightedLayerPooling
        self.config.update({"output_hidden_states": True})

        if pretrained:
            self.model = AutoModel.from_pretrained(CFG.model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Gradient Checkpointing
        if CFG.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Pooling
        # We use the number of layers specified in CFG
        self.weighted_pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers,
            layer_start=CFG.num_layers_pool,
        )

        # Head
        # Concatenated Mean + Max Pooling results in 2 * hidden_size
        self.fc = nn.Linear(self.config.hidden_size * 2, 1)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # 1. Weighted Layer Pooling (Aggregates across layers) -> (Batch, SeqLen, Hidden)
        weighted_embedding = self.weighted_pooler(all_hidden_states)

        # 2. Masking padding tokens
        # attention_mask shape: (Batch, SeqLen) -> unsqueeze to (Batch, SeqLen, 1)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(weighted_embedding.size()).float()
        )

        # 3. Mean Pooling
        sum_embeddings = torch.sum(weighted_embedding * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # 4. Max Pooling
        # Set padding tokens to large negative value so they aren't picked as max
        weighted_embedding_masked = weighted_embedding.clone()
        weighted_embedding_masked[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(weighted_embedding_masked, 1)[0]

        # 5. Concatenate
        concat_embeddings = torch.cat([mean_embeddings, max_embeddings], 1)

        return concat_embeddings

    def forward(self, input_ids, attention_mask, labels=None):
        # Extract features
        feature_vector = self.feature(input_ids, attention_mask)

        # Regression Head
        output = self.fc(feature_vector)

        return output.squeeze()
