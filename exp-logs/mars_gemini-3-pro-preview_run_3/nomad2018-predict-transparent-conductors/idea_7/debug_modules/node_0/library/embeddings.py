import os
import numpy as np
import pandas as pd
import torch
import ase.io
from tqdm import tqdm

# Import MatGL and Pymatgen components
try:
    import matgl
    from matgl.ext.ase import M3GNetCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    raise ImportError(
        "MatGL or Pymatgen not installed. Please ensure they are available."
    )

from library.config import Config
from library.data_manager import load_metadata
from library.descriptors import extract_descriptors

# Set random seeds
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)


class MatGLExtractor:
    """
    Feature extractor using a pre-trained MatGL (M3GNet) model.
    Extracts distributional node embeddings (Mean, Std, Range) for crystal structures.
    """

    def __init__(self, model_name=Config.MATGL_MODEL_NAME):
        """
        Initialize the MatGL model.
        """
        try:
            # Load the potential model
            self.potential = matgl.load_model(model_name)
            # Access the underlying M3GNet GNN
            self.model = self.potential.model
            self.element_types = self.model.element_types
            self.cutoff = self.model.cutoff

            # Initialize the graph converter
            # We use the same converter logic as the model uses internally
            from matgl.graph.converters import Structure2Graph

            self.converter = Structure2Graph(
                element_types=self.element_types, cutoff=self.cutoff
            )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"Error loading MatGL model: {e}")
            raise

    def process_structure(self, atoms):
        """
        Process a single ASE Atoms object to extract distributional embeddings.

        Args:
            atoms (ase.Atoms): The crystal structure.

        Returns:
            np.ndarray: A 1D array containing [Mean, Std, Range] of node embeddings.
        """
        try:
            # Convert ASE Atoms to Pymatgen Structure
            structure = AseAtomsAdaptor.get_structure(atoms)

            # Convert to graph
            graph, state_attr = self.converter.get_graph(structure)

            # Move to device
            graph = graph.to(self.device)
            state_attr = state_attr.to(self.device)

            # Extract features manually by running the model layers
            # 1. Embedding
            node_types = graph.ndata["node_type"]
            bond_dist = graph.edata["bond_dist"]

            node_feat = self.model.embedding(node_types)
            edge_feat = self.model.basis_expansion(bond_dist)

            # 2. Encoder (GNN Layers)
            # The encoder returns the updated node features
            node_feat = self.model.encoder(graph, edge_feat, node_feat, state_attr)

            # node_feat is (N_atoms, D)

            # 3. Distributional Pooling
            # Mean
            mean_feat = torch.mean(node_feat, dim=0)

            # Standard Deviation (use unbiased=False for consistency with numpy default, or True)
            # If N=1, std is 0.
            if node_feat.shape[0] > 1:
                std_feat = torch.std(node_feat, dim=0, unbiased=False)
            else:
                std_feat = torch.zeros_like(mean_feat)

            # Range (Max - Min)
            max_feat, _ = torch.max(node_feat, dim=0)
            min_feat, _ = torch.min(node_feat, dim=0)
            range_feat = max_feat - min_feat

            # Concatenate
            combined_feat = torch.cat([mean_feat, std_feat, range_feat], dim=0)

            return combined_feat.detach().cpu().numpy()

        except Exception as e:
            print(f"Error processing structure: {e}")
            # Return zero vector of appropriate size if failure
            # M3GNet embedding size is typically 64. 3 * 64 = 192.
            # We need to determine D dynamically or hardcode.
            # Default M3GNet dim is 64.
            dim = 64
            return np.zeros(dim * 3, dtype=np.float32)


def extract_features(split="train", sample_size=None, load_cached_data=True):
    """
    Orchestrates the extraction of all features (GNN + Physical + Tabular).
    Handles caching of the final combined dataset.

    Args:
        split (str): Dataset split ('train', 'val', 'test').
        sample_size (int, optional): Number of samples to process (for debugging).
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing all features and targets (if available).
    """
    # Determine cache path based on split
    if split == "train":
        cache_path = Config.TRAIN_FEATURES_PATH
    elif split == "val":
        cache_path = Config.VAL_FEATURES_PATH
    elif split == "test":
        cache_path = Config.TEST_FEATURES_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Handle sampling in cache filename to avoid collisions during debug
    if sample_size is not None:
        base, ext = os.path.splitext(cache_path)
        cache_path = f"{base}_sample_{sample_size}{ext}"

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached combined features from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Generating features for {split} set...")

    # Load metadata
    metadata_df = load_metadata(split=split, sample_size=sample_size)

    # A. Extract Physical Descriptors (Volume, Density, Bond Length)
    # This uses the provided library function which also caches internally
    print("Step A: Physical Descriptors")
    # We pass load_cached_data=False here to ensure we don't load a stale full-dataset cache
    # if we are currently subsampling, or we rely on the library to handle it.
    # To be safe and simple, we recompute for the specific metadata subset.
    physical_df = extract_descriptors(metadata_df, split=split, load_cached_data=False)

    # B. Extract GNN Embeddings
    print("Step B: GNN Distributional Embeddings")
    extractor = MatGLExtractor()

    gnn_features = []
    ids = []

    # Iterate through metadata to process files
    # Using tqdm for progress tracking is helpful but we'll keep it minimal
    for idx, row in metadata_df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            atoms = ase.io.read(file_path)
            feats = extractor.process_structure(atoms)
            gnn_features.append(feats)
            ids.append(row["id"])
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            # Append zeros matching the dimension (64*3 = 192)
            gnn_features.append(np.zeros(192, dtype=np.float32))
            ids.append(row["id"])

    # Convert to DataFrame
    gnn_features = np.array(gnn_features)
    gnn_cols = [f"gnn_{i}" for i in range(gnn_features.shape[1])]
    gnn_df = pd.DataFrame(gnn_features, columns=gnn_cols, index=metadata_df.index)

    # C. Combine All Features
    print("Step C: Merging Features")

    # Select tabular features from metadata
    # We exclude file_path. We keep ID for reference.
    # We also keep targets if they exist.
    tabular_cols = [c for c in metadata_df.columns if c != "file_path"]
    tabular_df = metadata_df[tabular_cols]

    # Concatenate: Tabular + Physical + GNN
    # Ensure indices align
    combined_df = pd.concat([tabular_df, physical_df, gnn_df], axis=1)

    # 3. Save to Cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        combined_df.to_parquet(cache_path, index=False)
        print(f"Saved combined features to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return combined_df
