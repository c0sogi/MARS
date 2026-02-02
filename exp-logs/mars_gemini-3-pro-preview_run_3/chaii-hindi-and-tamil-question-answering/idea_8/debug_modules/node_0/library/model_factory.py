import os
import torch
from transformers import XLMRobertaForTokenClassification, AutoConfig, logging
from library.config import Config

# Set transformers logging to error to suppress expected warnings when loading
# MLM weights into a TokenClassification architecture (head mismatch).
logging.set_verbosity_error()


def get_model(config: Config):
    """
    Initializes the XLM-Roberta model for Token Classification.

    Logic:
    1. Checks if a TAPT-finetuned model exists in the directory specified by config.tapt_output_dir.
    2. If found, loads weights from there to benefit from domain adaptation.
    3. If not found, loads the base pretrained model (xlm-roberta-base).
    4. Initializes the classification head with num_labels=3 (O, B-ANS, I-ANS).
    5. Moves the model to the appropriate device (GPU/CPU).

    Args:
        config (Config): Configuration object containing paths, model name, and device settings.

    Returns:
        XLMRobertaForTokenClassification: The instantiated PyTorch model.
    """

    # Check for TAPT artifacts (safetensors or bin)
    tapt_path = config.tapt_output_dir
    has_safetensors = os.path.exists(os.path.join(tapt_path, "model.safetensors"))
    has_bin = os.path.exists(os.path.join(tapt_path, "pytorch_model.bin"))

    model_path = config.model_name

    if has_safetensors or has_bin:
        print(f"Model Factory: Loading TAPT-finetuned weights from {tapt_path}")
        model_path = tapt_path
    else:
        print(
            f"Model Factory: TAPT weights not found. Loading base model {config.model_name}"
        )

    # Load Configuration
    # We explicitly set num_labels to 3 for the QA tagging task
    try:
        model_config = AutoConfig.from_pretrained(
            model_path, num_labels=config.num_labels
        )
    except Exception as e:
        print(
            f"Model Factory Error: Could not load config from {model_path}. Reverting to base. ({e})"
        )
        model_path = config.model_name
        model_config = AutoConfig.from_pretrained(
            model_path, num_labels=config.num_labels
        )

    # Instantiate Model
    # If loading from TAPT (MLM), the language modeling head is dropped and
    # a new token classification head is initialized.
    model = XLMRobertaForTokenClassification.from_pretrained(
        model_path, config=model_config
    )

    # Move to computation device
    model.to(config.device)

    return model
