import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig
from library.config import Config


class QATokenClassifier(nn.Module):
    """
    A wrapper class for a Hugging Face Transformer model for Token Classification.
    This model projects token embeddings into BIO tags (B-ANS, I-ANS, O) to extract
    answers from the retrieved context.
    """

    def __init__(self, model_name=Config.MODEL_NAME):
        """
        Initializes the QATokenClassifier.

        Args:
            model_name (str): The name or path of the pre-trained model to load.
                              Defaults to the value in Config.MODEL_NAME.
        """
        super(QATokenClassifier, self).__init__()

        # Configure the model with the specific number of labels and ID mappings
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=Config.NUM_LABELS,
            id2label=Config.IDS_TO_LABELS,
            label2id=Config.LABELS_TO_IDS,
        )

        # Load the pre-trained model for token classification
        # This automatically creates the classification head on top of the transformer
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, config=self.config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs the forward pass.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
                                      Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
                                           Shape: (batch_size, sequence_length)
            labels (torch.Tensor, optional): Labels for computing the token classification loss.
                                             Indices should be in [0, ..., config.num_labels - 1].
                                             Shape: (batch_size, sequence_length)

        Returns:
            transformers.modeling_outputs.TokenClassifierOutput:
                The output object containing `logits` (shape: batch_size, seq_len, num_labels)
                and `loss` (scalar) if labels are provided.
        """
        # Pass inputs to the underlying Hugging Face model
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return output

    def save(self, path):
        """
        Saves the model's state dictionary to the specified path.

        Args:
            path (str): The file path to save the model state (.pt or .pth).
        """
        torch.save(self.state_dict(), path)

    def load(self, path, device=Config.DEVICE):
        """
        Loads the model's state dictionary from the specified path.

        Args:
            path (str): The file path to load the model state from.
            device (torch.device): The device to map the model weights to.
        """
        self.load_state_dict(torch.load(path, map_location=device))
