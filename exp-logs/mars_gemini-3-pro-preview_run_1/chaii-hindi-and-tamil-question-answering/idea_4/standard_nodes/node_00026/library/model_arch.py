import os
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
    # Always load config from the base model definition to ensure correct architecture
    config = AutoConfig.from_pretrained(Config.MODEL_CHECKPOINT)

    # Initialize the model with the base architecture
    model = AutoModelForQuestionAnswering.from_pretrained(
        Config.MODEL_CHECKPOINT, config=config
    )

    # If a specific checkpoint path is provided, load those weights
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            # Load weights from the binary file (state_dict)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
        elif os.path.isdir(checkpoint_path):
            # If it's a directory, load using the transformers method
            model = AutoModelForQuestionAnswering.from_pretrained(
                checkpoint_path, config=config
            )

    # Move model to the computation device (GPU/CPU) defined in Config
    model.to(Config.DEVICE)

    return model
