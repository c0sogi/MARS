import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomTransformer(nn.Module):
    """
    A custom Transformer model for Author Identification that implements
    a Multi-Layer Concatenation Head.

    This architecture extracts the [CLS] tokens from the last four hidden layers
    of the backbone, concatenates them, and passes the result through a linear
    classification layer. This helps capture hierarchical stylistic features.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = Config.NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            model_name (str): The HuggingFace model identifier (e.g., 'roberta-base').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
            dropout_rate (float): Dropout probability for the classification head.
        """
        super(CustomTransformer, self).__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)
        # Ensure we can access intermediate layers
        self.config.output_hidden_states = True
        self.config.return_dict = True

        # Load backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Determine hidden size (usually 768 for base models)
        self.hidden_size = self.config.hidden_size

        # The head concatenates [CLS] tokens from the last 4 layers
        # Input dimension = hidden_size * 4
        self.concat_hidden_size = self.hidden_size * 4

        # Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.concat_hidden_size, num_classes)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (for some models).

        Returns:
            torch.Tensor: Logits for the classes.
        """
        # Pass inputs through the backbone
        # We use **kwargs to handle model-specific arguments safely if needed,
        # but explicit arguments are clearer.
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # outputs.hidden_states is a tuple of tensors (one for the output of the embeddings
        # + one for the output of each layer).
        # Shape of each hidden state: (batch_size, sequence_length, hidden_size)
        all_hidden_states = outputs.hidden_states

        # Extract the last 4 layers
        # We assume the tuple contains at least 4 layers + embeddings
        last_four_layers = all_hidden_states[-4:]

        # Extract the [CLS] token (index 0) from each of the last 4 layers
        # Shape of each: (batch_size, hidden_size)
        cls_embeddings = [layer[:, 0, :] for layer in last_four_layers]

        # Concatenate them along the feature dimension
        # Shape: (batch_size, hidden_size * 4)
        concatenated_cls = torch.cat(cls_embeddings, dim=-1)

        # Apply Dropout
        x = self.dropout(concatenated_cls)

        # Pass through Linear Classifier
        logits = self.fc(x)

        return logits
