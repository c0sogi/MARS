import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training.

    This utility perturbs the input embeddings during training to make the model
    robust to small variations in the input space, acting as a regularizer.
    """

    def __init__(self, model):
        """
        Initialize FGM.

        Args:
            model (nn.Module): The model to attack.
        """
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        """
        Applies adversarial perturbation to the embeddings.

        Args:
            epsilon (float): Magnitude of the perturbation.
            emb_name (str): Substring to identify embedding parameters in named_parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # Save original data
                self.backup[name] = param.data.clone()

                # Calculate norm of gradients
                norm = torch.norm(param.grad)

                # Apply perturbation if gradient is non-zero and valid
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        """
        Restores the original embeddings after the forward/backward pass.

        Args:
            emb_name (str): Substring to identify embedding parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class CustomXLMRoberta(nn.Module):
    """
    Custom XLM-RoBERTa model for Question Answering.

    Features:
    - Backbone: xlm-roberta-large
    - Span Head: Predicts start and end token probabilities.
    - Relevance Head: Predicts if the context window contains the answer.
    """

    def __init__(self, config: Config):
        """
        Initialize the model structure.

        Args:
            config (Config): Configuration object containing model parameters.
        """
        super(CustomXLMRoberta, self).__init__()
        self.config = config

        # Load HuggingFace configuration to retrieve hidden size
        hf_config = AutoConfig.from_pretrained(config.model_name)

        # Initialize Backbone
        self.backbone = AutoModel.from_pretrained(config.model_name, config=hf_config)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 1. Span Head (Start Logits + End Logits)
        # Input: Hidden Size -> Output: 2 (Start, End)
        self.qa_outputs = nn.Linear(hf_config.hidden_size, 2)

        # 2. Relevance Head (Binary Classification)
        # Input: Hidden Size -> Output: 1 (Logit for 'Is Answerable')
        self.relevance_classifier = nn.Linear(hf_config.hidden_size, 1)

        # Initialize weights for the new heads
        self._init_weights(self.qa_outputs)
        self._init_weights(self.relevance_classifier)

    def _init_weights(self, module):
        """
        Initialize weights for linear layers using a normal distribution.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.

        Returns:
            start_logits (torch.Tensor): Logits for the start position of the answer. Shape: (Batch, Seq_Len)
            end_logits (torch.Tensor): Logits for the end position of the answer. Shape: (Batch, Seq_Len)
            relevance_logits (torch.Tensor): Logits indicating if the answer is present. Shape: (Batch)
        """
        # Pass inputs through the backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Sequence output: Hidden states for all tokens (Batch, Seq_Len, Hidden)
        sequence_output = outputs.last_hidden_state

        # CLS token state: Representation of the first token (Batch, Hidden)
        # Note: XLM-R uses <s> as the first token (index 0)
        cls_token_state = sequence_output[:, 0, :]

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        cls_token_state = self.dropout(cls_token_state)

        # 1. Compute Span Logits
        # (Batch, Seq_Len, Hidden) -> (Batch, Seq_Len, 2)
        logits = self.qa_outputs(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Remove the last dimension: (Batch, Seq_Len, 1) -> (Batch, Seq_Len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 2. Compute Relevance Logits
        # (Batch, Hidden) -> (Batch, 1) -> (Batch)
        relevance_logits = self.relevance_classifier(cls_token_state).squeeze(-1)

        return start_logits, end_logits, relevance_logits
