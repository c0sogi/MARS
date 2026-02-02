import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from library.config import Config


class PatentModel(nn.Module):
    """
    DeBERTa-v3-Large Cross-Encoder with a Standard Classification Head.

    This model wraps the Hugging Face AutoModelForSequenceClassification.
    It relies on the standard regression head (Linear layer) provided by the library,
    avoiding complex regularization like MSD/WLP to ensure faster convergence
    as per the strategic plan.
    """

    def __init__(self, pretrained_model_name: str = Config.model_name):
        super().__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.config.num_labels = 1  # Regression task

        # Initialize model with standard regression head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, config=self.config
        )

        # Enable gradient checkpointing for memory efficiency with Large models
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass delegating to the HF model.
        Returns the standard SequenceClassifierOutput.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )


def get_optimizer_params(
    model: nn.Module, base_lr: float, weight_decay: float, llrd_decay: float
):
    """
    Constructs optimizer parameter groups with Layer-wise Learning Rate Decay (LLRD).

    Strategy:
    - Head (Classifier/Pooler): base_lr
    - Encoder Layer N: base_lr * (llrd_decay ^ 1)
    - ...
    - Embeddings: base_lr * (llrd_decay ^ max_depth)

    Args:
        model: The PatentModel instance.
        base_lr: The learning rate for the head (Config.learning_rate).
        weight_decay: Weight decay for regularization.
        llrd_decay: Multiplicative decay factor per layer (Config.llrd_decay).

    Returns:
        List of dictionaries containing parameter groups for the optimizer.
    """
    # Access the underlying Hugging Face model
    hf_model = model.model

    # Determine the number of layers (e.g., 24 for DeBERTa-Large)
    if hasattr(hf_model.config, "num_hidden_layers"):
        num_layers = hf_model.config.num_hidden_layers
    else:
        # Fallback for models that might use different config names
        num_layers = 24

    # Parameters to exclude from weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Initialize groups dictionary
    # Keys: Layer ID (0 to num_layers + 1)
    # Values: {'decay': [], 'no_decay': []}
    groups = {i: {"decay": [], "no_decay": []} for i in range(num_layers + 2)}

    for name, param in hf_model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine Layer ID based on parameter name
        if "embeddings" in name:
            # Embeddings are at the bottom
            layer_id = 0
        elif "encoder.layer" in name:
            # Extract layer index from name (e.g., "deberta.encoder.layer.15.output...")
            try:
                parts = name.split(".")
                # Find the segment after "layer"
                for i, part in enumerate(parts):
                    if (
                        part == "layer"
                        and i + 1 < len(parts)
                        and parts[i + 1].isdigit()
                    ):
                        # Layer 0 becomes ID 1, Layer 23 becomes ID 24
                        layer_id = int(parts[i + 1]) + 1
                        break
                else:
                    # Fallback if parsing fails
                    layer_id = 0
            except:
                layer_id = 0
        else:
            # Classifier, Pooler, and any top-level components go to the Head group
            layer_id = num_layers + 1

        # Assign to appropriate decay/no-decay list
        if any(nd in name for nd in no_decay):
            groups[layer_id]["no_decay"].append(param)
        else:
            groups[layer_id]["decay"].append(param)

    # Construct the final optimizer parameter list
    optimizer_parameters = []

    for layer_id in range(num_layers + 2):
        # Calculate depth from the head
        # Head (ID: num_layers+1) -> depth 0
        # Embeddings (ID: 0) -> depth num_layers+1
        depth = (num_layers + 1) - layer_id

        # Calculate Learning Rate for this group
        lr = base_lr * (llrd_decay**depth)

        # Add group with weight decay
        if groups[layer_id]["decay"]:
            optimizer_parameters.append(
                {
                    "params": groups[layer_id]["decay"],
                    "weight_decay": weight_decay,
                    "lr": lr,
                }
            )

        # Add group without weight decay
        if groups[layer_id]["no_decay"]:
            optimizer_parameters.append(
                {"params": groups[layer_id]["no_decay"], "weight_decay": 0.0, "lr": lr}
            )

    return optimizer_parameters
