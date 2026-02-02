import os
import pandas as pd
import ase.io
from library.config import INPUT_DIR, METADATA_DIR, RANDOM_SEED


class DataLoader:
    """
    Handles loading of metadata and geometry files.
    """

    def __init__(self):
        self.input_dir = INPUT_DIR
        self.metadata_dir = METADATA_DIR

    def load_metadata(self, split="train", debug=False, sample_size=50):
        """
        Loads the metadata CSV for a specific split.

        Args:
            split (str): One of 'train', 'val', 'test'.
            debug (bool): If True, returns a subsample of the data.
            sample_size (int): Number of samples to return in debug mode.

        Returns:
            pd.DataFrame: The loaded metadata.
        """
        file_name = f"{split}_metadata.csv"
        file_path = os.path.join(self.metadata_dir, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        df = pd.read_csv(file_path)

        if debug:
            # Deterministic sampling for reproducibility
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(
                    drop=True
                )
                print(
                    f"[DataLoader] Debug mode: Sampled {sample_size} rows from {split} set."
                )

        return df

    def load_geometry(self, relative_path):
        """
        Loads the geometry file (XYZ) using ASE.

        Args:
            relative_path (str): Path relative to the input directory (e.g., 'train/1/geometry.xyz').

        Returns:
            ase.Atoms: The atomic structure object, or None if loading fails.
        """
        full_path = os.path.join(self.input_dir, relative_path)

        if not os.path.exists(full_path):
            print(f"[DataLoader] Error: Geometry file not found at {full_path}")
            return None

        try:
            # ASE read can infer format automatically
            atoms = ase.io.read(full_path)
            return atoms
        except Exception as e:
            print(f"[DataLoader] Error reading geometry file {full_path}: {e}")
            return None

    def get_train_data(self, debug=False, sample_size=50):
        """Convenience method to get training data."""
        return self.load_metadata("train", debug=debug, sample_size=sample_size)

    def get_val_data(self, debug=False, sample_size=50):
        """Convenience method to get validation data."""
        return self.load_metadata("val", debug=debug, sample_size=sample_size)

    def get_test_data(self, debug=False, sample_size=50):
        """Convenience method to get test data."""
        return self.load_metadata("test", debug=debug, sample_size=sample_size)
