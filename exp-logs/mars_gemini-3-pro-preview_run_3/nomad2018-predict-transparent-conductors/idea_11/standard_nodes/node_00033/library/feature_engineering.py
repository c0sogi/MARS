import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from collections import Counter
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


class FeaturePipeline:
    """
    Orchestrates feature extraction, caching, and merging.
    """

    def __init__(self):
        self.phys_desc = PhysicalDescriptor()
        self.rdf_desc = RDFDescriptor()

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

            # Combine
            combined = {**phys_feats, **rdf_feats}
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
