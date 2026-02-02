import os
import sys
import random
import logging
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="idea_9"):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    # prevent adding multiple handlers to the same logger if called repeatedly
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(c_handler)

    return logger


def get_cpc_texts():
    """
    Returns a dictionary mapping CPC Section codes to their descriptions.

    The Cooperative Patent Classification (CPC) is hierarchical.
    The 'context' column in the dataset typically represents the Class level (e.g., 'A47').
    The first letter represents the Section.

    Mapping:
        A: Human Necessities
        B: Performing Operations; Transporting
        C: Chemistry; Metallurgy
        D: Textiles; Paper
        E: Fixed Constructions
        F: Mechanical Engineering; Lighting; Heating; Weapons; Blasting
        G: Physics
        H: Electricity
        Y: General Tagging of New Technological Developments

    Returns:
        dict: A dictionary where keys are Section codes and values are descriptions.
    """
    cpc_texts = {
        "A": "Human Necessities",
        "B": "Performing Operations; Transporting",
        "C": "Chemistry; Metallurgy",
        "D": "Textiles; Paper",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
        "G": "Physics",
        "H": "Electricity",
        "Y": "General Tagging of New Technological Developments",
    }
    return cpc_texts


def get_expanded_cpc_text(context_code, cpc_texts):
    """
    Resolves a specific context code (e.g., 'A47') to its description using the provided dictionary.

    Args:
        context_code (str): The CPC context code from the dataset.
        cpc_texts (dict): The dictionary returned by get_cpc_texts().

    Returns:
        str: The description associated with the Section of the context code.
    """
    if not context_code:
        return ""

    # The section is the first character of the context code
    section = context_code[0].upper()
    return cpc_texts.get(section, "")
