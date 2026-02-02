import torch
import torch.nn as nn
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel, XLMRobertaConfig


class CustomXLMR(XLMRobertaPreTrainedModel):
    """
    Custom XLM-R Large model for Question Answering with Multi-Task Learning.
    Includes:
    1. Span Head: Predicts start and end logits for the answer span.
    2. Relevance Head: Predicts whether the answer exists in the current window (CLS token).
    """

    config_class = XLMRobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config)

        # Span Head: Predicts start and end logits (2 outputs per token)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Relevance Head: Binary classification on [CLS] token (1 output)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights using the PreTrainedModel method
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices.

        Returns:
            dict: Contains 'start_logits', 'end_logits', and 'relevance_logits'.
        """
        # Pass through the backbone
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        sequence_output = outputs[0]  # (Batch, Seq_Len, Hidden)

        # Use the [CLS] token (first token) for relevance classification
        # In XLM-R, the first token is <s> which serves as CLS
        pooled_output = sequence_output[:, 0, :]  # (Batch, Hidden)

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Span Head Predictions
        logits = self.qa_outputs(sequence_output)  # (Batch, Seq_Len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (Batch, Seq_Len)
        end_logits = end_logits.squeeze(-1)  # (Batch, Seq_Len)

        # Relevance Head Prediction
        relevance_logits = self.classifier(pooled_output).squeeze(-1)  # (Batch,)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "relevance_logits": relevance_logits,
        }


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training.
    Perturbs input embeddings based on gradients to smooth the loss landscape.
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        """
        Adds perturbation to the embeddings.

        Args:
            epsilon (float): Magnitude of the perturbation.
            emb_name (str): Substring to identify embedding parameters in the model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # Save original parameters
                self.backup[name] = param.data.clone()

                # Calculate perturbation
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        """
        Restores the original embeddings.

        Args:
            emb_name (str): Substring to identify embedding parameters in the model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]

        self.backup = {}
