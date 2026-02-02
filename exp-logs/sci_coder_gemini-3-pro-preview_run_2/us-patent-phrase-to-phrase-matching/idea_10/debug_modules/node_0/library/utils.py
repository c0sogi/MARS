import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_pearson(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted scores.
    Handles potential dimension mismatches by flattening arrays.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Avoid division by zero if constant prediction
    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        return 0.0

    return np.corrcoef(y_true, y_pred)[0, 1]


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_cpc_texts(load_cached_data=True):
    """
    Retrieves the textual descriptions for CPC context codes.
    Implements caching using Parquet files to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        pd.DataFrame: DataFrame with columns ['context', 'context_text']
    """
    cache_path = Config.cpc_text_path

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # 2. Generate data if cache miss or force reload
    # Since we do not have external internet access or the specific cpc-data dataset file
    # in the input list, we define a dictionary of common CPC codes relevant to the dataset
    # (based on the provided analysis and general domain knowledge of the CPC scheme).

    cpc_data = {
        "A": "Human Necessities",
        "B": "Performing Operations; Transporting",
        "C": "Chemistry; Metallurgy",
        "D": "Textiles; Paper",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
        "G": "Physics",
        "H": "Electricity",
        "Y": "General Tagging of New Technological Developments",
        # Specific codes observed in analysis
        "A47": "Furniture; Domestic Articles or Appliances; Coffee Mills; Spice Mills; Suction Cleaners",
        "A61": "Medical or Veterinary Science; Hygiene",
        "C23": "Coating Metallic Material; Coating Material with Metallic Material",
        "E03": "Water Supply; Sewerage",
        "F15": "Fluid-Pressure Actuators; Hydraulics or Pneumatics in General",
        "F16": "Engineering Elements or Units; General Measures for Producing and Maintaining Effective Functioning of Machines or Installations; Thermal Insulation in General",
        "G01": "Measuring; Testing",
        "G03": "Photography; Cinematography; Analogous Techniques using Waves other than Optical Waves; Electrography; Holography",
        "H01": "Basic Electric Elements",
        "H04": "Electric Communication Technique",
    }

    # We need to ensure we cover all contexts present in the data.
    # We load train/test to get the list of unique contexts.
    train_df = pd.read_csv(Config.train_path)
    test_df = pd.read_csv(Config.test_path)
    val_df = pd.read_csv(Config.val_path)

    all_contexts = pd.concat(
        [train_df["context"], val_df["context"], test_df["context"]]
    ).unique()

    data = []
    for ctx in all_contexts:
        description = cpc_data.get(ctx)

        # Fallback logic if exact code not in dictionary
        if description is None:
            # Try to map based on section (first letter)
            section = ctx[0]
            section_desc = cpc_data.get(section, "Technical Domain")
            description = f"{section_desc} - {ctx}"

        data.append({"context": ctx, "context_text": description})

    df = pd.DataFrame(data)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df
