import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from collections import Counter
import torch
import warnings

# Import configuration and utils
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    RDF_CUTOFF,
    RDF_BINS,
    ELEMENTS,
    MATGL_MODEL_NAME,
    RANDOM_SEED,
)
from library.data_utils import load_metadata, read_geometry

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class PhysicalDescriptor:
    """
    Extracts explicit physical properties from the atomic structure.
    """

    def calculate(self, atoms: ase.Atoms) -> dict:
        try:
            vol = atoms.get_volume()
            # Density in atomic units (sum of masses / volume)
            mass = sum(atoms.get_masses())
            density = mass / vol if vol > 0 else 0.0
            return {
                "phys_volume": vol,
                "phys_density": density,
                "phys_num_atoms": len(atoms),
            }
        except Exception:
            return {"phys_volume": 0.0, "phys_density": 0.0, "phys_num_atoms": 0}


class RDFDescriptor:
    """
    Computes element-resolved Radial Distribution Functions (RDF).
    """

    def __init__(self, cutoff=RDF_CUTOFF, n_bins=RDF_BINS, elements=ELEMENTS):
        self.cutoff = cutoff
        self.n_bins = n_bins
        self.elements = sorted(elements)
        # Create pairs: (Al, Al), (Al, Ga), ... including self-pairs
        self.pairs = []
        for i in range(len(self.elements)):
            for j in range(i, len(self.elements)):
                self.pairs.append((self.elements[i], self.elements[j]))

        # Define bin edges
        self.bins = np.linspace(0, cutoff, n_bins + 1)

    def calculate(self, atoms: ase.Atoms) -> dict:
        features = {}

        # Get neighbor list with distances
        # i: atom indices, j: neighbor indices, d: distances
        try:
            i_indices, j_indices, d_dists = neighbor_list("ijd", atoms, self.cutoff)
        except Exception:
            # Fallback for empty or failed calculation
            for p1, p2 in self.pairs:
                for b in range(self.n_bins):
                    features[f"rdf_{p1}_{p2}_{b}"] = 0.0
            return features

        symbols = np.array(atoms.get_chemical_symbols())

        # Group distances by element pair
        # We only consider i < j to avoid double counting, or handle full symmetric
        # neighbor_list returns both i->j and j->i.
        # For RDF, we usually want the distribution of distances per pair type.

        # Optimization: Map symbols to integers for faster indexing
        sym_to_int = {s: k for k, s in enumerate(self.elements)}
        # Filter atoms not in our list (just in case)
        valid_mask = np.array([s in sym_to_int for s in symbols])

        if not valid_mask.all():
            # If there are unexpected elements, we just ignore them or handle gracefully
            pass

        # Process each pair type
        for el1, el2 in self.pairs:
            # Find indices of el1 and el2
            # We look for pairs where symbol[i] == el1 and symbol[j] == el2
            # Since neighbor list has both directions, we can just look for source==el1 and dest==el2
            # But we need to be careful about double counting for same-element pairs.
            # neighbor_list returns all neighbors j for i.

            mask_src = symbols[i_indices] == el1
            mask_dst = symbols[j_indices] == el2
            mask = mask_src & mask_dst

            dists = d_dists[mask]

            # Histogram
            hist, _ = np.histogram(dists, bins=self.bins)

            # Normalize by volume or number of atoms to make it intensive?
            # Standard RDF is normalized by density, but raw counts/volume is often a good descriptor for ML.
            # Normalizing by cell volume helps invariance to system size.
            vol = atoms.get_volume()
            if vol > 0:
                hist = hist / vol

            for b in range(self.n_bins):
                features[f"rdf_{el1}_{el2}_{b}"] = hist[b]

        return features


class MatGLEmbedder:
    """
    Extracts implicit structural embeddings using a pre-trained M3GNet model.
    """

    def __init__(self, model_name=MATGL_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.graph_converter = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hook_features = {}
        self._load_model()

    def _load_model(self):
        try:
            import matgl
            from matgl.ext.ase import AseAtomsAdaptor

            # Load model
            # This might download the model. If it fails (no internet), we handle it.
            self.model = matgl.load_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # The graph converter is usually part of the model pipeline or we use the adaptor
            # M3GNet takes a graph. We use the converter from the loaded model if available,
            # or standard one.
            # In matgl 1.1.3, we typically convert atoms -> structure -> graph
            self.adaptor = AseAtomsAdaptor()

            # Register hook to get the embedding before the final readout
            # Inspecting M3GNet structure: usually has a 'readout' or final MLP
            # We will try to hook into the last layer of the GNN or the input to readout
            # Common attribute for readout in M3GNet is `readout`
            if hasattr(self.model, "readout"):
                self.model.readout.register_forward_hook(self._hook_fn)
            else:
                # Fallback: try to hook the last layer of the model if possible
                # or just use the model output if it's a vector (unlikely for PES)
                pass

        except Exception as e:
            print(
                f"Warning: Failed to load MatGL model ({str(e)}). MatGL features will be zeros."
            )
            self.model = None

    def _hook_fn(self, module, input, output):
        # input is a tuple, usually (node_feats, ...) or just node_feats
        # We want the graph-level embedding.
        # M3GNet readout usually takes (g, node_feats, ...).
        # If the hook is on the readout module, 'input' might be the features going INTO the readout.
        # We'll assume input[0] contains the features we want to aggregate or use.
        # However, capturing 'input' to readout is tricky if it involves the graph structure.
        # Let's try to capture 'output' of the GNN part if possible.
        # Simpler approach: The 'output' of the readout is the final property (scalar).
        # We want the latent vector.
        # Let's assume input[0] is the feature tensor.
        try:
            if isinstance(input, tuple) and len(input) > 0:
                self.hook_features["embedding"] = input[0]
            else:
                self.hook_features["embedding"] = input
        except:
            pass

    def calculate(self, atoms: ase.Atoms) -> dict:
        # Default zero vector size (M3GNet usually has 64 or 128 dim)
        embedding_dim = 64
        features = {f"matgl_{i}": 0.0 for i in range(embedding_dim)}

        if self.model is None:
            return features

        try:
            import matgl

            # Convert ASE atoms to Pymatgen Structure
            struct = self.adaptor.get_structure(atoms)

            # Convert to graph
            # Note: The exact API depends on matgl version.
            # Assuming model.graph_converter exists or we use the model's predict method logic
            # For robustness, we try to use the model's internal converter if possible
            # or just use `matgl.graph.converters.GraphConverter`

            # A robust way in matgl is to use the model directly if it accepts structure
            # But M3GNet usually takes graphs.
            # Let's try to use the graph converter associated with the model
            if hasattr(self.model, "graph_converter"):
                graph, state = self.model.graph_converter.get_graph(struct)
            else:
                # Attempt to find a default converter
                from matgl.graph.converters import Pmg2Graph

                # Default cutoff for M3GNet is usually 5.0
                converter = Pmg2Graph(
                    element_types=self.model.element_types, cutoff=5.0
                )
                graph, state = converter.get_graph(struct)

            graph = graph.to(self.device)
            state = torch.tensor(state).to(self.device)

            # Forward pass
            # We need to clear previous hook features
            self.hook_features = {}

            with torch.no_grad():
                # M3GNet forward signature: g, state_attr=None, l_g=None
                # We pass the graph.
                _ = self.model(graph, state_attr=state)

            # Retrieve embedding from hook
            # The input to readout is usually (g, node_feats).
            # We want a graph-level embedding. If node_feats are passed, we need to pool them.
            # If the hook captured node features (num_atoms, dim), we mean pool.

            emb = self.hook_features.get("embedding")

            # If emb is a tuple (g, h), extract h
            if isinstance(emb, tuple):
                for item in emb:
                    if isinstance(item, torch.Tensor):
                        emb = item
                        break

            if isinstance(emb, torch.Tensor):
                # If shape is (num_atoms, dim), mean pool
                if emb.dim() == 2 and emb.shape[0] == len(atoms):
                    emb = torch.mean(emb, dim=0)

                # If shape is (1, dim) or (dim,)
                emb = emb.cpu().numpy().flatten()

                # Update features
                for i in range(min(len(emb), embedding_dim)):
                    features[f"matgl_{i}"] = float(emb[i])

        except Exception as e:
            # print(f"MatGL inference failed: {e}")
            pass

        return features


class FeaturePipeline:
    """
    Orchestrates feature extraction, caching, and merging.
    """

    def __init__(self):
        self.phys_desc = PhysicalDescriptor()
        self.rdf_desc = RDFDescriptor()
        self.matgl_desc = MatGLEmbedder()

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inner loop to compute features for a dataframe.
        """
        all_features = []

        print(f"Extracting features for {len(df)} samples...")

        for idx, row in df.iterrows():
            # Read geometry
            try:
                atoms = read_geometry(row["file_path"])
            except Exception:
                # If file missing or corrupt, create dummy atoms or skip?
                # We'll create a dummy empty atoms object to generate zero features
                atoms = ase.Atoms()

            # 1. Physical
            phys_feats = self.phys_desc.calculate(atoms)

            # 2. RDF
            rdf_feats = self.rdf_desc.calculate(atoms)

            # 3. MatGL
            matgl_feats = self.matgl_desc.calculate(atoms)

            # Combine
            combined = {**phys_feats, **rdf_feats, **matgl_feats}
            all_features.append(combined)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)}")

        return pd.DataFrame(all_features)

    def process_split(
        self, split: str, sample_size: int = None, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Main method to process a data split (train/val/test).
        Handles loading metadata, checking cache, computing features, and saving cache.
        """
        # Determine cache path
        if split == "train":
            cache_path = TRAIN_FEATURES_PATH
        elif split == "val":
            cache_path = VAL_FEATURES_PATH
        elif split == "test":
            cache_path = TEST_FEATURES_PATH
        else:
            raise ValueError("Unknown split")

        # Load Metadata
        df_meta = load_metadata(split, sample_size)

        # Check cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            try:
                df_features = pd.read_parquet(cache_path)

                # Verify length matches (in case sample_size changed or cache is stale)
                if len(df_features) == len(df_meta):
                    # Merge and return
                    # We assume index alignment. Reset index to be safe.
                    df_meta = df_meta.reset_index(drop=True)
                    df_features = df_features.reset_index(drop=True)

                    # Concatenate features to metadata
                    # Drop columns that might duplicate if re-merged (though simple concat is usually fine)
                    df_final = pd.concat([df_meta, df_features], axis=1)
                    return df_final
                else:
                    print(
                        f"Cache length mismatch ({len(df_features)} vs {len(df_meta)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Compute Features
        print(f"Computing features for {split} split...")
        df_features = self._compute_features(df_meta)

        # Save Cache
        print(f"Saving features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        # Merge
        df_meta = df_meta.reset_index(drop=True)
        df_features = df_features.reset_index(drop=True)
        df_final = pd.concat([df_meta, df_features], axis=1)

        return df_final
