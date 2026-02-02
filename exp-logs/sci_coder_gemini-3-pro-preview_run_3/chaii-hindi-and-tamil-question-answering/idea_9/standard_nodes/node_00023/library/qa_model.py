import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.configuration import Config


class XLMRobertaForQA(nn.Module):
    """
    XLMRoberta-based Token Classification model for Question Answering.

    This model wraps the XLM-Roberta backbone and adds a linear classification head
    to predict one of 3 classes for each token:
    0: O (Outside)
    1: B-ANS (Beginning of Answer)
    2: I-ANS (Inside of Answer)
    """

    def __init__(self, pretrained_path=None):
        """
        Initializes the model.

        Args:
            pretrained_path (str, optional): Path to a pretrained checkpoint directory
                                             (e.g., the output of TAPT). If None, loads
                                             the base model defined in Config.MODEL_CHECKPOINT.
        """
        super(XLMRobertaForQA, self).__init__()

        # specific configuration for the model
        model_name = pretrained_path if pretrained_path else Config.MODEL_CHECKPOINT
        print(f"Initializing XLMRobertaForQA backbone from: {model_name}")

        # Load configuration to ensure hidden sizes match
        self.config = AutoConfig.from_pretrained(model_name)

        # Load the backbone model
        # add_pooling_layer=False is used because we only need the sequence output
        # for token classification, not the pooled sentence embedding.
        self.roberta = AutoModel.from_pretrained(
            model_name, config=self.config, add_pooling_layer=False
        )

        # Token Classification Head
        self.num_labels = 3
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        # Loss Function
        # ignore_index=-100 ensures we don't compute loss for padding or special tokens
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
                                      Shape: (Batch Size, Sequence Length)
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
                                           Shape: (Batch Size, Sequence Length)
            labels (torch.Tensor, optional): Labels for computing the token classification loss.
                                             Shape: (Batch Size, Sequence Length)

        Returns:
            tuple: (loss, logits) if labels are provided.
            tuple: (logits,) if labels are not provided.
        """
        # Pass inputs through the backbone
        # outputs[0] is the sequence_output (last_hidden_state)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        # Apply classifier
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # Shape: (Batch, Seq_Len, 3)

        output = (logits,)

        if labels is not None:
            # Flatten logits and labels to compute loss over all tokens in the batch
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            output = (loss,) + output

        return output
