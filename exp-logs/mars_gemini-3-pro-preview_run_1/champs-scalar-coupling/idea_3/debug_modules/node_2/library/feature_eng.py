import os
import numpy as np
import pandas as pd
from library.config import GASEConfig
from library.data_utils import process_and_cache_data
from library.mpnn_trainer import MPNNRunner


class FeatureAssembler:
    """
    Assembles the hybrid feature set by combining explicit geometric features
    (from tabular data) with learned interaction embeddings (from MPNN).
    """

    def __init__(self):
        self.embedding_paths = {
            "train": GASEConfig.EMBEDDINGS_TRAIN_PATH,
            "val": GASEConfig.EMBEDDINGS_VAL_PATH,
            "test": GASEConfig.EMBEDDINGS_TEST_PATH,
        }

    def _ensure_embeddings_exist(self, split, load_cached_data):
        """
        Checks if embedding file exists. If not, attempts to generate it using MPNNRunner.
        """
        path = self.embedding_paths[split]

        # If cache load is requested and file exists, we are good
        if load_cached_data and os.path.exists(path):
            return

        # Otherwise, or if file missing, we need to generate
        print(
            f"[{split}] Embeddings not found or regeneration requested. Generating..."
        )

        # We use the MPNNRunner to extract embeddings
        # This assumes the MPNN model has been trained and saved.
        # If the model is missing, MPNNRunner will raise FileNotFoundError.
        runner = MPNNRunner()

        # MPNNRunner.extract_embeddings generates for ALL splits at once.
        runner.extract_embeddings(load_cached_data=load_cached_data)

        if not os.path.exists(path):
            raise RuntimeError(
                f"Failed to generate embeddings at {path} after running extraction."
            )

    def assemble_data(self, split, load_cached_data=True):
        """
        Loads tabular data and embeddings, merges them, and returns the combined DataFrame.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached intermediate files.

        Returns:
            pd.DataFrame: Combined dataframe with geometric features and embedding columns.
        """
        # 1. Load Tabular Data (Geometric Features)
        # process_and_cache_data handles its own caching logic and returns all splits
        df_train, df_val, df_test = process_and_cache_data(
            load_cached_data=load_cached_data
        )

        if split == "train":
            df = df_train
        elif split == "val":
            df = df_val
        elif split == "test":
            df = df_test
        else:
            raise ValueError(f"Unknown split: {split}")

        # 2. Load Embeddings
        self._ensure_embeddings_exist(split, load_cached_data)
        try:
            embeddings = np.load(self.embedding_paths[split])
        except Exception as e:
            raise IOError(f"Error loading embeddings for {split}: {e}")

        # 3. Validate Alignment
        if len(df) != len(embeddings):
            raise ValueError(
                f"Data mismatch for split '{split}': "
                f"Tabular rows={len(df)}, Embeddings rows={len(embeddings)}. "
                "Ensure both were generated from the same metadata."
            )

        # 4. Merge
        print(
            f"[{split}] Merging {embeddings.shape[1]} embedding features into DataFrame..."
        )

        # Generate column names for embeddings
        embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]

        # Create DataFrame for embeddings to facilitate concat
        # We reset index of df to ensure alignment with the 0-indexed numpy array
        df = df.reset_index(drop=True)
        df_embed = pd.DataFrame(embeddings, columns=embed_cols)

        # Concatenate
        df_combined = pd.concat([df, df_embed], axis=1)

        return df_combined


def prepare_stratified_data(df):
    """
    Partitions the dataframe by coupling type.

    Args:
        df (pd.DataFrame): The combined dataframe containing 'type' column.

    Returns:
        dict: A dictionary mapping coupling type (str) to DataFrame subset.
    """
    stratified_data = {}

    # Ensure type column exists
    if "type" not in df.columns:
        raise KeyError("DataFrame must contain 'type' column for stratification.")

    # Group by type
    # We iterate through the groups provided by pandas
    for coupling_type, group in df.groupby("type"):
        # We store a copy to avoid SettingWithCopy warnings downstream
        stratified_data[coupling_type] = group.copy().reset_index(drop=True)

    return stratified_data
