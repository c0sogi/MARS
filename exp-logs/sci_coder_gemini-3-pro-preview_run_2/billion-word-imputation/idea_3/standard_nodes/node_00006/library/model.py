import torch
import torch.nn as nn
import numpy as np
import random
from transformers import AutoModel, AutoConfig
from library.config import Config


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Set fixed random seeds immediately upon module import
set_seed(Config.SEED)


class DualHeadTransformer(nn.Module):
    """
    Transformer-Based Dual-Head Sequence Tagger for Sentence Infilling.

    This model uses a pre-trained Transformer encoder to generate contextual embeddings
    for each token in the input sequence. It then branches into two heads:
    1. Location Head: Predicts the probability that the missing word belongs immediately
       after the current token.
    2. Word Head: Predicts the identity of the missing word from a target vocabulary.
    """

    def __init__(
        self,
        vocab_size,
        model_name=Config.MODEL_NAME,
        hidden_size=Config.HIDDEN_SIZE,
        dropout_prob=Config.DROPOUT,
    ):
        """
        Initialize the DualHeadTransformer.

        Args:
            vocab_size (int): The size of the target word vocabulary (including special tokens).
            model_name (str): The Hugging Face model identifier for the backbone.
            hidden_size (int): The hidden dimension size of the backbone output.
            dropout_prob (float): The dropout probability applied to the hidden states.
        """
        super(DualHeadTransformer, self).__init__()

        # Load the pre-trained backbone configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_prob)

        # ----------------------------------------------------------------------
        # Head 1: Location Prediction
        # ----------------------------------------------------------------------
        # Projects hidden state to a scalar logit.
        # High value indicates the missing word should be inserted AFTER this token.
        self.location_head = nn.Linear(hidden_size, 1)

        # ----------------------------------------------------------------------
        # Head 2: Word Prediction
        # ----------------------------------------------------------------------
        # Projects hidden state to the target vocabulary size.
        # Predicts which word is missing, assuming this is the correct location.
        self.word_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (Batch, Seq_Len).
            attention_mask (torch.Tensor): Attention mask of shape (Batch, Seq_Len).

        Returns:
            tuple: (loc_logits, word_logits)
                - loc_logits (torch.Tensor): Shape (Batch, Seq_Len). Unnormalized scores
                  for the insertion location.
                - word_logits (torch.Tensor): Shape (Batch, Seq_Len, Vocab_Size).
                  Unnormalized scores for the missing word prediction.
        """
        # 1. Backbone Encoding
        # outputs.last_hidden_state shape: (Batch, Seq_Len, Hidden)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # 2. Location Head
        # Output shape: (Batch, Seq_Len, 1) -> Squeeze to (Batch, Seq_Len)
        loc_logits = self.location_head(sequence_output)
        loc_logits = loc_logits.squeeze(-1)

        # 3. Word Head
        # Output shape: (Batch, Seq_Len, Vocab_Size)
        word_logits = self.word_head(sequence_output)

        return loc_logits, word_logits
