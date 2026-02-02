import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler


def calculate_cell_volume(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    Calculates the volume of a unit cell given lattice parameters.

    Args:
        a, b, c (float): Lattice vector lengths.
        alpha_deg, beta_deg, gamma_deg (float): Lattice angles in degrees.

    Returns:
        float: The volume of the unit cell.
    """
    alpha_rad = np.radians(alpha_deg)
    beta_rad = np.radians(beta_deg)
    gamma_rad = np.radians(gamma_deg)

    term = (
        1
        - np.cos(alpha_rad) ** 2
        - np.cos(beta_rad) ** 2
        - np.cos(gamma_rad) ** 2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    # Ensure non-negative value inside sqrt due to floating point errors
    volume = a * b * c * np.sqrt(np.maximum(0.0, term))
    return volume


def extract_global_features(row):
    """
    Extracts macroscopic properties from a metadata row.

    Args:
        row (pd.Series or dict): Row containing material metadata.

    Returns:
        np.ndarray: A 1D array of shape (11,) containing:
            - Lattice lengths (3)
            - Lattice angles (3)
            - Unit cell volume (1)
            - Atomic density (1)
            - Stoichiometry (3)
    """
    # Extract lattice parameters
    a = row["lattice_vector_1_ang"]
    b = row["lattice_vector_2_ang"]
    c = row["lattice_vector_3_ang"]
    alpha = row["lattice_angle_alpha_degree"]
    beta = row["lattice_angle_beta_degree"]
    gamma = row["lattice_angle_gamma_degree"]

    # Calculate volume
    volume = calculate_cell_volume(a, b, c, alpha, beta, gamma)

    # Calculate atomic density
    n_atoms = row["number_of_total_atoms"]
    # Avoid division by zero
    density = n_atoms / volume if volume > 1e-9 else 0.0

    # Extract stoichiometry
    stoich = [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]]

    features = np.array(
        [a, b, c, alpha, beta, gamma, volume, density, stoich[0], stoich[1], stoich[2]],
        dtype=np.float32,
    )

    return features


def process_symmetry(row):
    """
    Extracts the Spacegroup ID.

    Args:
        row (pd.Series or dict): Row containing material metadata.

    Returns:
        int: Spacegroup ID.
    """
    return int(row["spacegroup"])


class FeatureScaler:
    """
    Manages the standardization of continuous features.
    Wraps sklearn.preprocessing.StandardScaler.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            X (np.ndarray): The data used to compute the mean and standard deviation.
        """
        self.scaler.fit(X)
        self.is_fitted = True

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Args:
            X (np.ndarray): The data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before calling transform.")
        return self.scaler.transform(X)

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        """
        Save the scaler state to a file.

        Args:
            path (str): File path.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)

    def load(self, path):
        """
        Load the scaler state from a file.

        Args:
            path (str): File path.
        """
        if os.path.exists(path):
            self.scaler = joblib.load(path)
            self.is_fitted = True
        else:
            raise FileNotFoundError(f"Scaler file not found at {path}")
