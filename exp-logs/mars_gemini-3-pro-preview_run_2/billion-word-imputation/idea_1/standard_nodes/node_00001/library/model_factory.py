import os
import random
import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from typing import Tuple, Optional
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed: The integer seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def load_model_and_tokenizer(
    model_name: str = Config.MODEL_NAME,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads the pre-trained model and tokenizer, and moves the model to the configured device.

    Args:
        model_name: The name or path of the pre-trained model to load.
                    Defaults to Config.MODEL_NAME.

    Returns:
        A tuple containing (model, tokenizer).
    """
    # Ensure seeds are set before initialization
    set_seed()

    device = Config.get_device()

    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print(f"Loading model for: {model_name}")
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    print(f"Moving model to device: {device}")
    model.to(device)

    return model, tokenizer
