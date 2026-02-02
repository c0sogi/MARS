import os
import random
import numpy as np
import torch
from library.config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_genus(species_name):
    """
    Parses the genus name from a full species label.
    Assumes the format 'Genus_Species' or 'Genus_Species_Subspecies'.

    Args:
        species_name (str): The full species name (e.g., 'Acer_Capillipes').

    Returns:
        str: The extracted genus name (e.g., 'Acer').
    """
    if not isinstance(species_name, str):
        # Handle potential non-string inputs gracefully
        return str(species_name).split("_")[0]
    return species_name.split("_")[0]


def get_species_to_genus_mapping(species_le, genus_le):
    """
    Generates a mapping array where the index corresponds to the species class ID
    and the value corresponds to the genus class ID.

    This is used for the taxonomic Bayesian update step:
    P(Species|Features) * P(Genus|Features) -> Refined Probability.

    Args:
        species_le (LabelEncoder): Fitted sklearn LabelEncoder for species targets.
        genus_le (LabelEncoder): Fitted sklearn LabelEncoder for genus targets.

    Returns:
        np.ndarray: An array of shape (n_species,) where array[i] is the genus ID for species ID i.
    """
    # 1. Get the ordered list of species names corresponding to indices 0..N-1
    # species_le.classes_ is sorted and corresponds to the integer transform
    species_names = species_le.classes_

    # 2. Extract the genus for each species name
    genus_names = [extract_genus(name) for name in species_names]

    # 3. Transform these genus names into their integer IDs using the genus encoder
    # This results in an array where index 'i' (species ID) contains 'j' (genus ID)
    species_to_genus_indices = genus_le.transform(genus_names)

    return species_to_genus_indices
