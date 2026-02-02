import torch
from transformers import AutoModelForQuestionAnswering, AutoConfig
from library.config import Config


def get_model(checkpoint_path=None):
    """
    Initializes and returns the MuRIL-based Question Answering model.

    This function handles loading both the base pre-trained weights (for training start)
    and fine-tuned checkpoints (for inference or resuming). It utilizes the
    AutoModelForQuestionAnswering architecture which adds a linear classification head
    for span prediction (start and end logits).

    Args:
        checkpoint_path (str, optional): Path to a local checkpoint directory or file.
                                         If None, loads the base 'google/muril-base-cased'
                                         defined in Config.MODEL_CHECKPOINT.

    Returns:
        model (AutoModelForQuestionAnswering): The PyTorch model moved to the configured device.
    """
    # Determine the model path to load
    model_name_or_path = checkpoint_path if checkpoint_path else Config.MODEL_CHECKPOINT

    # Load the configuration
    # This ensures that architecture specific parameters are correctly loaded
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Initialize the model
    # AutoModelForQuestionAnswering automatically creates the architecture for
    # extractive QA (Encoder + Linear Head for 2 outputs: start_logits, end_logits)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path, config=config
    )

    # Move model to the computation device (GPU/CPU) defined in Config
    model.to(Config.DEVICE)

    return model
