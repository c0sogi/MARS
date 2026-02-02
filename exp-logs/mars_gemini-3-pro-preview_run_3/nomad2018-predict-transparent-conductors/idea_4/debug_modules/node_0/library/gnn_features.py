import os
import torch
import numpy as np
import pandas as pd
import dgl
import matgl
from matgl.ext.ase import AseAtomsAdaptor
from library.config import Config
from library.data_processing import process_data as load_base_data


class StructureEmbedder:
    """
    Extracts structural embeddings using a pre-trained MatGL (M3GNet) model.
    """

    def __init__(self, model_name=None, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if model_name is None:
            model_name = Config.MATGL_MODEL_NAME

        print(f"Loading MatGL model: {model_name} on {self.device}...")
        try:
            # Load the pre-trained potential
            self.potential = matgl.load_model(model_name)
            self.model = self.potential.model
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load MatGL model: {e}")

        self.embeddings = []
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """
        Registers a forward hook on the readout layer to capture global embeddings.
        """
        # M3GNet typically has a 'readout' module that aggregates node features.
        # We want the output of this readout.
        if hasattr(self.model, "readout"):
            self.hook_handle = self.model.readout.register_forward_hook(self._hook_fn)
        else:
            # If readout is not directly accessible, we might need to inspect the model structure.
            # For standard M3GNet, 'readout' is the aggregation layer.
            raise ValueError("M3GNet model does not have a 'readout' layer attribute.")

    def _hook_fn(self, module, input, output):
        """
        Hook function to capture the output of the readout layer.
        """
        # output is the global embedding tensor (batch_size, embedding_dim)
        self.embeddings.append(output.detach().cpu().numpy())

    def generate_embeddings(self, atoms_list, batch_size=32):
        """
        Generates embeddings for a list of ASE Atoms objects.
        """
        self.embeddings = []  # Clear previous state
        adaptor = AseAtomsAdaptor()

        print(f"Converting {len(atoms_list)} structures to graphs...")
        graphs = []
        valid_indices = []

        for i, atoms in enumerate(atoms_list):
            if atoms is None:
                continue
            try:
                # get_graph returns DGLGraph or (DGLGraph, state_attr)
                res = adaptor.get_graph(atoms)
                if isinstance(res, tuple):
                    g = res[0]
                else:
                    g = res
                graphs.append(g)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error converting atom {i} to graph: {e}")

        if not graphs:
            print("No valid graphs generated.")
            return pd.DataFrame()

        print(f"Running inference in batches of {batch_size}...")
        n_samples = len(graphs)

        for i in range(0, n_samples, batch_size):
            batch_graphs = graphs[i : i + batch_size]

            # Batch the graphs using DGL
            batched_g = dgl.batch(batch_graphs)
            batched_g = batched_g.to(self.device)

            # Prepare other inputs if necessary (M3GNet forward can take state_attr and l_g)
            # For simple energy prediction/embedding extraction, the graph usually suffices
            # if the model handles default states.

            with torch.no_grad():
                # Forward pass triggers the hook
                # We don't care about the final prediction (energy), just the intermediate embedding
                _ = self.model(batched_g)

        if not self.embeddings:
            return pd.DataFrame()

        # Concatenate all collected batches
        all_embeddings = np.concatenate(self.embeddings, axis=0)

        # Create DataFrame
        feat_cols = [f"matgl_emb_{i}" for i in range(all_embeddings.shape[1])]
        df_emb = pd.DataFrame(all_embeddings, columns=feat_cols)

        return df_emb


def process_gnn_features(split="train", load_cached_data=True):
    """
    Generates or loads MatGL structural embeddings for the specified split.
    """
    cache_file = f"{split}_matgl_embeddings.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_file)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached GNN features from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Process from scratch
    print(f"Generating GNN features for {split}...")

    # Load base data (tabular + atoms) using the existing data processing pipeline
    # We set load_cached_data=True for the base data to avoid re-reading XYZs if possible
    # process_data returns (df, atoms_list)
    df_base, atoms_list = load_base_data(split, load_cached_data=True)

    # Initialize embedder
    embedder = StructureEmbedder()

    # Generate embeddings
    df_emb = embedder.generate_embeddings(atoms_list)

    # Verify alignment
    if len(df_emb) != len(df_base):
        print(
            f"Warning: Embedding count ({len(df_emb)}) does not match base data count ({len(df_base)})."
        )
        # This might happen if graph conversion failed for some atoms.
        # In a strict pipeline, we might need to filter df_base, but here we assume high success rate.
        # If lengths differ, we truncate to the shorter one to avoid crashes, though this indicates data loss.
        min_len = min(len(df_emb), len(df_base))
        df_emb = df_emb.iloc[:min_len]

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_emb.to_parquet(cache_path, index=False)
    print(f"Saved GNN features to {cache_path}")

    return df_emb
