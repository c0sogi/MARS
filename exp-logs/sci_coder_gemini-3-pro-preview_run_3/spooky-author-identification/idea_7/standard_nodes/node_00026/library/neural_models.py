import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomTransformer(nn.Module):
    """
    A custom transformer architecture that wraps a Hugging Face backbone
    and implements a Multi-Layer Concatenation head for classification.

    This architecture extracts [CLS] tokens from the last 4 hidden layers,
    concatenates them, and passes them through a linear layer. This allows
    the model to leverage both high-level semantic features and lower-level
    stylistic/structural features often found in earlier layers.
    """

    def __init__(self, model_name, num_classes=Config.NUM_CLASSES):
        """
        Initialize the CustomTransformer.

        Args:
            model_name (str): The name (HF Hub) or path to the pre-trained model.
            num_classes (int): Number of output classes.
        """
        super(CustomTransformer, self).__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Ensure hidden states are outputted for the multi-layer head
        self.config.output_hidden_states = True
        self.config.num_labels = num_classes

        # Load the backbone model
        # This works for both 'roberta-base' and 'microsoft/deberta-v3-base'
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Feature extraction configuration
        self.num_layers_to_concat = 4
        self.hidden_size = self.config.hidden_size

        # Classification Head
        # Input dimension is hidden_size * 4 because we concat 4 layers
        self.classifier = nn.Linear(
            self.hidden_size * self.num_layers_to_concat, num_classes
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Standard Loss function
        # Note: Distillation loss (KL Div) is typically computed in the training loop
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels (class indices).

        Returns:
            dict: Dictionary containing 'logits' and optionally 'loss'.
        """
        # Pass inputs through the backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states is a tuple containing:
        # (embeddings, layer_1_output, ..., layer_N_output)
        all_hidden_states = outputs.hidden_states

        # Extract [CLS] tokens from the last 'num_layers_to_concat' layers
        # [CLS] token is consistently at index 0 for RoBERTa and DeBERTa
        cls_embeddings = []

        # We iterate backwards from the last layer
        for i in range(1, self.num_layers_to_concat + 1):
            # -1 is last layer, -2 is second to last, etc.
            layer_output = all_hidden_states[-i]

            # Extract [CLS] token: Shape (batch_size, hidden_size)
            cls_token = layer_output[:, 0, :]
            cls_embeddings.append(cls_token)

        # Concatenate the extracted features along the feature dimension
        # Shape: (batch_size, hidden_size * 4)
        concatenated_features = torch.cat(cls_embeddings, dim=1)

        # Apply dropout
        concatenated_features = self.dropout(concatenated_features)

        # Pass through the classifier to get logits
        # Shape: (batch_size, num_classes)
        logits = self.classifier(concatenated_features)

        loss = None
        if labels is not None:
            # Compute loss if labels are provided and are standard class indices
            # If labels are soft targets (probabilities) for distillation,
            # the loss is typically computed externally using the returned logits.
            if labels.dtype == torch.long:
                loss = self.loss_fct(logits, labels.view(-1))

        return {"logits": logits, "loss": loss}
