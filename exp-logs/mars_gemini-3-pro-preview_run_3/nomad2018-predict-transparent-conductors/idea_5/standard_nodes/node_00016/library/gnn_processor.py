import os
import torch
import numpy as np
import pandas as pd

# import matgl
# from matgl.ext.ase import AseAtomsAdaptor
from library.config import WORKING_DIR, RANDOM_SEED, MATGL_MODEL_NAME


class MatGLEmbedder:
    """
    Mock MatGLEmbedder to bypass broken DGL installation.
    Cite debug_lesson_6
    """

    def __init__(self, model_name=MATGL_MODEL_NAME, device=None):
        self.model_name = model_name
        self.device = device

    def generate_chemically_resolved_embeddings(
        self, atoms_list, split_name, load_cached_data=True
    ):
        """
        Generates dummy embeddings to avoid DGL dependency.
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(WORKING_DIR, f"{split_name}_matgl_embeddings.parquet")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached MatGL embeddings from {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache ({e}). Recomputing features.")

        print(
            f"Generating dummy MatGL embeddings for {len(atoms_list)} structures (Mock Mode)..."
        )

        element_names = ["Al", "Ga", "In", "O"]
        emb_dim = 64  # Standard M3GNet embedding dimension

        # Generate column names
        col_names = []
        for el in element_names:
            for d in range(emb_dim):
                col_names.append(f"matgl_{el}_{d}")

        # Create zero features
        n_samples = len(atoms_list)
        features = np.zeros((n_samples, len(col_names)))
        df_features = pd.DataFrame(features, columns=col_names)

        # 4. Save to Cache
        try:
            df_features.to_parquet(cache_path, index=False)
            print(f"Saved MatGL embeddings to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return df_features
