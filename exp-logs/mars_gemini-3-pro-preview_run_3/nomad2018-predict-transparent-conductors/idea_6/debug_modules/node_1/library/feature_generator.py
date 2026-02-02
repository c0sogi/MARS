import os
import numpy as np
import pandas as pd
import torch
import ase.io
from tqdm import tqdm
import warnings

# Import configuration and data loader
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_COMBINED_FEATURES_PATH,
    VAL_COMBINED_FEATURES_PATH,
    TEST_COMBINED_FEATURES_PATH,
    ATOMIC_SPECIES,
    INPUT_DIR,
    MATGL_MODEL_NAME,
    TABULAR_FEATURES,
    RANDOM_SEED,
)
from library.data_loader import load_metadata, load_geometry

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class PhysicalDescriptorExtractor:
    """
    Extracts explicit physical descriptors from the atomic geometry.
    Calculates volume, density, and validates atom counts.
    """

    def __init__(self):
        # Atomic masses in atomic mass units (u)
        self.atomic_masses = {"Al": 26.981539, "Ga": 69.723, "In": 114.818, "O": 15.999}

    def extract(self, atoms: ase.Atoms) -> dict:
        """
        Calculates physical properties from an ASE Atoms object.
        """
        if atoms is None:
            return {"volume": np.nan, "density": np.nan, "num_atoms_geometry": np.nan}

        # 1. Volume (Angstrom^3)
        try:
            volume = atoms.get_volume()
        except ValueError:
            # Fallback for non-periodic systems or errors
            volume = np.nan

        # 2. Mass and Density
        # Density units: u / A^3. To convert to g/cm^3, multiply by 1.66054
        total_mass = sum(atoms.get_masses())
        if volume > 0:
            density = total_mass / volume
        else:
            density = np.nan

        # 3. Number of atoms
        num_atoms = len(atoms)

        return {"volume": volume, "density": density, "num_atoms_geometry": num_atoms}


class ChemicallyResolvedEmbedder:
    """
    Extracts chemically resolved structural embeddings using a pre-trained MatGL model.
    Implements Element-wise Statistical Pooling.
    """

    def __init__(self):
        self.model = None
        self.converter = None
        self.layer_output = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Loads the pre-trained MatGL model and sets up the graph converter.
        """
        try:
            import matgl
            from matgl.ext.ase import Atoms2Graph

            # Load the pre-trained potential
            # This might download the model if not present.
            # We assume the environment allows this or it's cached.
            self.model = matgl.load_model(MATGL_MODEL_NAME)

            # Setup converter
            # We need to ensure the converter uses the same element types as the model
            # M3GNet usually has an element_refs attribute or similar in the underlying model
            if hasattr(self.model.model, "element_refs"):
                elements = self.model.model.element_refs
            else:
                # Fallback to standard Materials Project elements if attribute missing
                # or just the subset we care about if the model supports it
                elements = None  # Atoms2Graph handles this if None usually

            self.converter = Atoms2Graph(element_types=elements, cutoff=5.0)

            # Register hook to capture node embeddings before readout
            # M3GNet architecture: Embedding -> GraphLayers -> Readout
            # We want the output of the last GraphLayer

            # Helper to capture output
            def hook_fn(module, input, output):
                # output of GraphLayer is usually the graph itself with updated features
                self.layer_output = output

            # Attach hook to the last graph layer
            # self.model.model.graph_layers is a ModuleList
            if (
                hasattr(self.model.model, "graph_layers")
                and len(self.model.model.graph_layers) > 0
            ):
                self.model.model.graph_layers[-1].register_forward_hook(hook_fn)
            else:
                print(
                    "Warning: Could not find graph_layers in MatGL model. Embedding extraction may fail."
                )

        except ImportError:
            print(
                "Error: matgl or dgl not installed. Embedding extraction will return NaNs."
            )
            self.model = None
        except Exception as e:
            print(f"Error initializing MatGL model: {e}")
            self.model = None

    def extract(self, atoms: ase.Atoms) -> dict:
        """
        Generates chemically resolved embeddings for the given structure.
        """
        # Define species of interest and embedding dimension (usually 64 for M3GNet)
        species_list = ATOMIC_SPECIES  # ['Al', 'Ga', 'In', 'O']
        embedding_dim = 64

        # Initialize result dictionary with NaNs
        features = {}
        for spec in species_list:
            for stat in ["mean", "std"]:
                for i in range(embedding_dim):
                    features[f"emb_{spec}_{stat}_{i}"] = np.nan

        if self.model is None or atoms is None:
            return features

        try:
            # Convert ASE atoms to DGL graph
            # The converter returns (graph, state_attr)
            graph, state_attr = self.converter.get_graph(atoms)

            # Run forward pass (inference mode)
            with torch.no_grad():
                # We only need to run the model to trigger the hook
                # The output of the model (energy/forces) is ignored
                _ = self.model(graph, state_attr)

            # Retrieve captured graph from hook
            if self.layer_output is None:
                return features

            # Extract node features from the graph
            # In DGL, node features are stored in ndata['node_feat']
            # The captured 'output' from the hook should be the DGL graph
            g_out = self.layer_output
            node_feats = g_out.ndata["node_feat"].cpu().numpy()  # Shape: (N_atoms, Dim)

            # Determine embedding dimension dynamically if possible
            actual_dim = node_feats.shape[1]

            # Get atomic symbols to map nodes to species
            symbols = np.array(atoms.get_chemical_symbols())

            # Element-wise Pooling
            for spec in species_list:
                # Find indices of atoms of this species
                indices = np.where(symbols == spec)[0]

                if len(indices) > 0:
                    # Extract specific embeddings
                    spec_embeddings = node_feats[indices]

                    # Compute stats
                    mean_vec = np.mean(spec_embeddings, axis=0)
                    std_vec = np.std(spec_embeddings, axis=0)

                    # Populate dictionary
                    # Ensure we handle dimension mismatch if model dim != 64
                    dim_to_use = min(embedding_dim, actual_dim)

                    for i in range(dim_to_use):
                        features[f"emb_{spec}_mean_{i}"] = mean_vec[i]
                        features[f"emb_{spec}_std_{i}"] = std_vec[i]

            # Cleanup hook capture for next run
            self.layer_output = None

        except Exception as e:
            # print(f"Error during embedding extraction: {e}")
            pass

        return features


def generate_features(
    split: str, load_cached_data: bool = True, limit: int = None
) -> pd.DataFrame:
    """
    Main function to generate the feature matrix for a given data split.
    Combines metadata, physical descriptors, and GNN embeddings.
    Handles caching to avoid re-computation.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        limit (int): Optional limit for debugging.

    Returns:
        pd.DataFrame: The complete feature matrix (X) combined with targets (y) if available.
    """
    # Determine cache path
    if split == "train":
        cache_path = TRAIN_COMBINED_FEATURES_PATH
    elif split == "val":
        cache_path = VAL_COMBINED_FEATURES_PATH
    elif split == "test":
        cache_path = TEST_COMBINED_FEATURES_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        df = pd.read_parquet(cache_path)
        if limit:
            return df.head(limit)
        return df

    print(f"Generating features for {split} set (Cache miss or force reload)...")

    # 2. Load Metadata
    meta_df = load_metadata(split)
    if limit:
        meta_df = meta_df.head(limit)

    # 3. Initialize Extractors
    phys_extractor = PhysicalDescriptorExtractor()
    emb_extractor = ChemicallyResolvedEmbedder()

    # 4. Iterate and Compute
    phys_features_list = []
    emb_features_list = []

    # Use tqdm for progress tracking
    print("Extracting physical and embedding features...")
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        # Load geometry
        atoms = load_geometry(row["file_path"])

        # Extract Physical Features
        phys_feats = phys_extractor.extract(atoms)
        phys_features_list.append(phys_feats)

        # Extract Embeddings
        emb_feats = emb_extractor.extract(atoms)
        emb_features_list.append(emb_feats)

    # 5. Create DataFrames
    phys_df = pd.DataFrame(phys_features_list, index=meta_df.index)
    emb_df = pd.DataFrame(emb_features_list, index=meta_df.index)

    # 6. Combine All Features
    # Start with tabular features from metadata
    # Ensure we only keep relevant tabular columns + targets + id
    cols_to_keep = ["id"] + TABULAR_FEATURES

    # Add targets if they exist (train/val)
    targets_exist = all(
        col in meta_df.columns
        for col in ["formation_energy_ev_natom", "bandgap_energy_ev"]
    )
    if targets_exist:
        cols_to_keep.extend(["formation_energy_ev_natom", "bandgap_energy_ev"])

    # Filter metadata
    base_df = meta_df[cols_to_keep].copy()

    # Concatenate horizontally
    final_df = pd.concat([base_df, phys_df, emb_df], axis=1)

    # 7. Save to Cache
    print(f"Saving generated features to {cache_path}...")
    # Ensure directory exists (redundant but safe)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    final_df.to_parquet(cache_path, index=False)

    return final_df
