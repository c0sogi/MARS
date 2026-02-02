import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from library.config import Config


class PhraseModel(nn.Module):
    """
    PhraseModel wraps a Hugging Face AutoModelForSequenceClassification.
    It is configured for a regression task (num_labels=1) to predict similarity scores.
    """

    def __init__(self, model_name=Config.model_name, config_path=None, pretrained=True):
        """
        Args:
            model_name (str): Name of the pre-trained model backbone.
            config_path (str, optional): Path to a local config file/directory.
            pretrained (bool): Whether to load pre-trained weights.
        """
        super().__init__()

        # Load Configuration
        if config_path:
            self.config = AutoConfig.from_pretrained(config_path)
        else:
            self.config = AutoConfig.from_pretrained(
                model_name, num_labels=Config.num_labels
            )

        # Enforce Regression Settings
        self.config.num_labels = 1
        self.config.attention_probs_dropout_prob = Config.attention_dropout
        self.config.hidden_dropout_prob = Config.dropout

        # Load Model
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=self.config
            )
        else:
            self.model = AutoModelForSequenceClassification.from_config(self.config)

        # Enable Gradient Checkpointing for memory efficiency with Large models
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices.
            labels (torch.Tensor, optional): Labels for computing the loss.

        Returns:
            SequenceClassifierOutput: Object containing loss and logits.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )


def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, layer_decay):
    """
    Constructs parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    The learning rate for a layer is calculated as:
        lr_layer = learning_rate * (layer_decay ^ (num_layers - layer_index))

    - The Classifier/Head gets the base 'learning_rate'.
    - The top transformer layer gets 'learning_rate * layer_decay'.
    - The bottom transformer layer gets 'learning_rate * layer_decay^N'.
    - Embeddings get the lowest learning rate.

    Args:
        model (nn.Module): The model to optimize.
        learning_rate (float): The base learning rate (for the head).
        weight_decay (float): Weight decay coefficient.
        layer_decay (float): Multiplicative decay factor per layer.

    Returns:
        list: A list of dictionaries defining parameter groups.
    """
    # Access the underlying HF model if wrapped in PhraseModel
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        hf_model = model.model
    else:
        hf_model = model

    # Get total number of layers (e.g., 24 for DeBERTa-Large)
    num_layers = hf_model.config.num_hidden_layers

    # Define parameters to exclude from weight decay
    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}

    # Dictionary to group parameters by (lr, weight_decay)
    # Key: (lr, weight_decay), Value: list of parameters
    grouped_params = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. Determine Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # 2. Determine Layer ID for LLRD
        # Parameter names typically look like:
        # - model.deberta.embeddings.word_embeddings.weight
        # - model.deberta.encoder.layer.0.attention...
        # - model.classifier.weight

        layer_id = -1  # Default to embeddings/bottom

        if "encoder.layer" in name:
            # Extract the layer index
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layer" and i + 1 < len(parts):
                    if parts[i + 1].isdigit():
                        layer_id = int(parts[i + 1])
                        break
        elif "embeddings" in name:
            layer_id = -1
        elif "classifier" in name or "pooler" in name:
            layer_id = num_layers
        # Note: 'rel_embeddings' or other components not in 'layer' blocks
        # are treated as bottom-level (-1) or similar to embeddings.

        # 3. Calculate Learning Rate Scaling
        # Scale = 0 for Head (highest LR)
        # Scale increases as we go deeper (lower LR)
        if layer_id == num_layers:
            scale = 0
        elif layer_id == -1:
            scale = num_layers + 1
        else:
            scale = num_layers - layer_id

        lr = learning_rate * (layer_decay**scale)

        # 4. Add to Group
        key = (lr, wd)
        if key not in grouped_params:
            grouped_params[key] = []
        grouped_params[key].append(param)

    # Convert to list format expected by Optimizer
    optimizer_groups = []
    for (lr, wd), params in grouped_params.items():
        optimizer_groups.append({"params": params, "lr": lr, "weight_decay": wd})

    return optimizer_groups
