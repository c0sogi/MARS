import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig
from library.config import Config


def get_model():
    """
    Loads the pre-trained MuRIL model for Question Answering.

    Returns:
        transformers.models.bert.modeling_bert.BertForQuestionAnswering:
        The MuRIL model initialized with weights from the checkpoint defined in Config.
    """
    print(f"Loading model from checkpoint: {Config.MODEL_CHECKPOINT}")

    # Load configuration from the checkpoint
    config = AutoConfig.from_pretrained(Config.MODEL_CHECKPOINT)

    # Initialize the model for Question Answering
    # This adds a linear layer on top of the hidden-states output to compute
    # start_logits and end_logits
    model = AutoModelForQuestionAnswering.from_pretrained(
        Config.MODEL_CHECKPOINT, config=config
    )

    return model


def get_tokenizer():
    """
    Loads the tokenizer associated with the MuRIL model.

    Returns:
        transformers.models.bert.tokenization_bert_fast.BertTokenizerFast:
        The tokenizer initialized from the checkpoint defined in Config.
    """
    print(f"Loading tokenizer from checkpoint: {Config.MODEL_CHECKPOINT}")

    # Load the tokenizer
    # We use the fast tokenizer implementation for efficiency
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    return tokenizer
