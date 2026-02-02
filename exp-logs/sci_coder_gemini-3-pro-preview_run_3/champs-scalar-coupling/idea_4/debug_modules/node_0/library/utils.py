import os
import json
import numpy as np
import pandas as pd
import torch
from library.config import Config

# ==========================================
# Geometry Utilities
# ==========================================


def compute_bond_angle(pos_i, pos_j, pos_k, epsilon=1e-7):
    """
    Computes the cosine of the angle between atoms i, j, k centered at j.

    Args:
        pos_i (torch.Tensor): Coordinates of atom i (N, 3)
        pos_j (torch.Tensor): Coordinates of atom j (N, 3) - the center atom
        pos_k (torch.Tensor): Coordinates of atom k (N, 3)
        epsilon (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Cosine of the angles (N,) in range [-1, 1].
    """
    # Vectors from center j
    vec_ji = pos_i - pos_j
    vec_jk = pos_k - pos_j

    # Norms
    norm_ji = torch.norm(vec_ji, dim=1)
    norm_jk = torch.norm(vec_jk, dim=1)

    # Dot product
    dot_product = torch.sum(vec_ji * vec_jk, dim=1)

    # Cosine calculation
    # Clamp denominator to avoid division by zero
    denominator = norm_ji * norm_jk
    denominator = torch.clamp(denominator, min=epsilon)

    cos_angle = dot_product / denominator

    # Clamp result to valid cosine range to handle numerical noise
    return torch.clamp(cos_angle, -1.0, 1.0)


# ==========================================
# Data Loading & Parsing Utilities
# ==========================================


def load_structures_table(path=Config.STRUCTURES_CSV):
    """
    Loads the structures CSV file into a Pandas DataFrame.

    Args:
        path (str): Path to structures.csv.

    Returns:
        pd.DataFrame: DataFrame containing atomic coordinates.
    """
    return pd.read_csv(path)


def parse_xyz_file(file_path):
    """
    Parses a single XYZ structure file.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        dict: {'atoms': np.array of symbols, 'coords': np.array of shape (N, 3)}
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    # line 1 is comment/energy

    atoms = []
    coords = []

    for line in lines[2 : 2 + num_atoms]:
        parts = line.strip().split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])

    return {"atoms": np.array(atoms), "coords": np.array(coords, dtype=np.float32)}


def map_atom_types(atom_symbols):
    """
    Maps atomic symbols to integer indices.
    Mapping: H=0, C=1, N=2, O=3, F=4

    Args:
        atom_symbols (np.array or list): List of atomic symbols.

    Returns:
        np.array: Integer indices.
    """
    mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    # Use list comprehension for speed on string lists, then convert to numpy
    return np.array([mapping.get(s, -1) for s in atom_symbols], dtype=np.int64)


def get_molecule_structure(structures_df, molecule_name):
    """
    Retrieves the structure data for a specific molecule from the global dataframe.

    Args:
        structures_df (pd.DataFrame): The dataframe loaded via load_structures_table.
        molecule_name (str): The name of the molecule.

    Returns:
        dict: {'atoms': np.array, 'coords': np.array}
    """
    mol_data = structures_df[structures_df["molecule_name"] == molecule_name]
    atoms = mol_data["atom"].values
    coords = mol_data[["x", "y", "z"]].values.astype(np.float32)
    return {"atoms": atoms, "coords": coords}


# ==========================================
# Target Standardization
# ==========================================


class TargetStandardizer:
    """
    Handles per-coupling-type standardization (Z-score normalization) of target variables.
    """

    def __init__(self):
        self.stats = {}
        self.coupling_types = Config.COUPLING_TYPES

    def fit(self, df):
        """
        Computes mean and std for each coupling type in the training DataFrame.

        Args:
            df (pd.DataFrame): Training dataframe with 'type' and 'scalar_coupling_constant'.
        """
        for c_type in self.coupling_types:
            subset = df[df["type"] == c_type]
            if len(subset) > 0:
                mean_val = float(subset["scalar_coupling_constant"].mean())
                std_val = float(subset["scalar_coupling_constant"].std())
            else:
                # Fallback if type is missing (unlikely in this dataset)
                mean_val = 0.0
                std_val = 1.0

            self.stats[c_type] = {"mean": mean_val, "std": std_val}

    def transform(self, df):
        """
        Standardizes the 'scalar_coupling_constant' column in the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing 'type' and 'scalar_coupling_constant'.

        Returns:
            np.array: Standardized target values.
        """
        # Create mapping series
        means = df["type"].map(lambda x: self.stats[x]["mean"])
        stds = df["type"].map(lambda x: self.stats[x]["std"])

        values = df["scalar_coupling_constant"].values
        return (values - means.values) / stds.values

    def inverse_transform(self, predictions, types):
        """
        Converts standardized predictions back to the original physical scale.

        Args:
            predictions (torch.Tensor or np.array): Predicted standardized values.
            types (np.array or list): Corresponding coupling types for each prediction.

        Returns:
            np.array: Predictions in original scale.
        """
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()

        if isinstance(types, list):
            types = np.array(types)

        restored = np.zeros_like(predictions, dtype=np.float32)

        # Vectorized restoration per type
        for c_type in self.coupling_types:
            if c_type in self.stats:
                mask = types == c_type
                if np.any(mask):
                    mean = self.stats[c_type]["mean"]
                    std = self.stats[c_type]["std"]
                    restored[mask] = predictions[mask] * std + mean

        return restored

    def save(self, directory):
        """
        Saves the statistics to a JSON file.

        Args:
            directory (str): Directory to save the file.
        """
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "target_stats.json")
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=4)

    def load(self, directory):
        """
        Loads statistics from a JSON file.

        Args:
            directory (str): Directory containing the file.
        """
        path = os.path.join(directory, "target_stats.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.stats = json.load(f)
        else:
            print(
                f"Warning: Stats file not found at {path}. Standardizer not initialized."
            )
