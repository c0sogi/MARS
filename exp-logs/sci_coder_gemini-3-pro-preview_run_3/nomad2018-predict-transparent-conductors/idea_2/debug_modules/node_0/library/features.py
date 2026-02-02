import os
import numpy as np
import pandas as pd
import torch
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor
import matgl
import dgl
from library.config import Config

# Ensure reproducible results
np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)


def compute_physical_descriptors(atoms):
    """
    Computes physical descriptors for a given ASE atoms object.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: A dictionary containing 'volume', 'density', and 'num_atoms'.
    """
    try:
        # Volume in Angstrom^3
        vol = atoms.get_volume()

        # Number of atoms
        n_atoms = len(atoms)

        # Density (Atomic Mass Units / Angstrom^3)
        # 1 AMU/A^3 approx 1.66 g/cm^3. We use the raw value as a feature.
        mass = sum(atoms.get_masses())
        density = mass / vol if vol > 1e-6 else 0.0

        return {"volume": vol, "density": density, "num_atoms": n_atoms}
    except Exception as e:
        print(f"Error computing descriptors: {e}")
        return {"volume": 0.0, "density": 0.0, "num_atoms": 0}


class GNNFeatureExtractor:
    """
    Extracts structural embeddings using a pre-trained M3GNet model.
    """

    def __init__(self, model_name=Config.MATGL_MODEL_NAME, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading M3GNet model '{model_name}' on {self.device}...")

        # Load the potential and extract the underlying GNN model
        self.potential = matgl.load_model(model_name)
        self.model = self.potential.model
        self.model.to(self.device)
        self.model.eval()

        # Graph converter from the potential
        self.graph_converter = self.potential.graph_converter

    def extract_features(self, atoms_list, batch_size=Config.GNN_BATCH_SIZE):
        """
        Converts a list of ASE atoms to graphs and extracts embeddings.

        Args:
            atoms_list (list): List of ASE Atoms objects.
            batch_size (int): Batch size for inference.

        Returns:
            np.ndarray: Array of shape (n_samples, embedding_dim).
        """
        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
        embeddings = []

        print(f"Extracting GNN features for {len(structures)} structures...")

        # Process in batches
        for i in range(0, len(structures), batch_size):
            batch_structs = structures[i : i + batch_size]
            graphs = []
            state_attrs = []
            l_graphs = []

            for struct in batch_structs:
                # Convert structure to graph
                # get_graph returns (graph, state_attr, line_graph)
                g, state, l_g = self.graph_converter.get_graph(struct)
                graphs.append(g)
                state_attrs.append(state)
                l_graphs.append(l_g)

            # Batch the graphs using DGL
            batched_g = dgl.batch(graphs).to(self.device)
            batched_state = torch.stack(state_attrs).to(self.device)
            batched_l_g = (
                dgl.batch(l_graphs).to(self.device) if l_graphs[0] is not None else None
            )

            with torch.no_grad():
                # Forward pass through the M3GNet model
                # The model updates node features in the graph 'g' in-place or returns output
                # We are interested in the node features before the final readout
                _ = self.model(g=batched_g, state_attr=batched_state, l_g=batched_l_g)

                # Extract node features. M3GNet typically stores them in "node_feat"
                # If "node_feat" is not present, we check for "h" or similar, but "node_feat" is standard.
                if "node_feat" in batched_g.ndata:
                    node_feats = batched_g.ndata["node_feat"]
                else:
                    # Fallback or error handling
                    # Try to find the feature key
                    keys = list(batched_g.ndata.keys())
                    if len(keys) > 0:
                        node_feats = batched_g.ndata[keys[0]]
                    else:
                        raise ValueError("No node features found in GNN output.")

                # Aggregate node features to graph level (Readout)
                # We use mean pooling to get a fixed-size vector per crystal
                graph_embeddings = dgl.readout_nodes(batched_g, node_feats, op="mean")

                embeddings.append(graph_embeddings.cpu().numpy())

        if not embeddings:
            return np.empty((0, 0))

        return np.concatenate(embeddings, axis=0)


def process_data(metadata_path, cache_path, load_cached_data=True):
    """
    Main data processing function. Loads metadata, extracts features (physical + GNN),
    and returns a combined DataFrame. Implements caching.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_path (str): Path to save/load the Parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed feature matrix.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            print(f"Loaded {len(df)} rows from cache.")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Metadata
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Debugging: Sample subset if configured
    if Config.DEBUG_SAMPLE_SIZE is not None:
        print(f"DEBUG: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()

    # 3. Extract Features
    # Lists to store results
    physical_feats = []
    atoms_objects = []
    valid_indices = []

    print("Reading geometry files and computing physical descriptors...")
    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            try:
                atoms = ase.io.read(full_path)

                # Physical descriptors
                phys = compute_physical_descriptors(atoms)
                physical_feats.append(phys)

                # Store for GNN
                atoms_objects.append(atoms)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
        else:
            print(f"File not found: {full_path}")

    # Filter dataframe to valid rows
    df_valid = df.loc[valid_indices].reset_index(drop=True)

    # Create DataFrame from physical features
    df_phys = pd.DataFrame(physical_feats)

    # 4. GNN Feature Extraction
    gnn_extractor = GNNFeatureExtractor()
    gnn_embeddings = gnn_extractor.extract_features(atoms_objects)

    # Create DataFrame from embeddings
    embedding_cols = [f"gnn_emb_{i}" for i in range(gnn_embeddings.shape[1])]
    df_gnn = pd.DataFrame(gnn_embeddings, columns=embedding_cols)

    # 5. Combine All Features
    # Concatenate: [Metadata (Tabular) + Physical + GNN]
    # Drop file_path as it's not a feature
    df_final = pd.concat(
        [df_valid.drop(columns=["file_path"]), df_phys, df_gnn], axis=1
    )

    # 6. Save to Cache
    print(f"Saving {len(df_final)} processed rows to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_final.to_parquet(cache_path, index=False)

    return df_final
