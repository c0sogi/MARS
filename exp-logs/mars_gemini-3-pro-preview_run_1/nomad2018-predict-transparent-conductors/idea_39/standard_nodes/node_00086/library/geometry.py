import os
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from ase import Atoms
from library.config import Config


class GeometryProcessor:
    def __init__(self, atom_map=Config.ATOM_MAP, k_neighbors=Config.K_NEIGHBORS):
        self.atom_map = atom_map
        self.k_neighbors = k_neighbors
        self.num_atom_types = len(atom_map)
        # Mapping from atomic number to feature index
        # Al: 13, Ga: 31, In: 49, O: 8
        self.z_to_idx = {13: 0, 31: 1, 49: 2, 8: 3}

    def parse_xyz(self, file_path):
        """
        Parses an XYZ file using ASE, with a manual fallback for specific formats.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Attempt standard ASE read
            atoms = read(file_path)
            # Check if cell is populated (sometimes custom formats don't load cell automatically)
            if np.all(atoms.cell.lengths() == 0):
                raise ValueError("ASE read returned zero cell")
        except Exception:
            # Fallback to manual parsing for the competition specific format
            atoms = self._manual_parse_xyz(file_path)

        return atoms

    def _manual_parse_xyz(self, file_path):
        """
        Manual parser for .xyz files with 'lattice_vector' lines.
        """
        symbols = []
        positions = []
        cell = []

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                cell.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                # Format: atom x y z symbol
                positions.append([float(x) for x in parts[1:4]])
                symbols.append(parts[4])

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        return atoms

    def get_k_nearest_neighbors(self, atoms):
        """
        Finds the K nearest neighbors for each atom respecting PBC.
        Returns:
            neighbor_indices: (N, K) array of indices
            neighbor_distances: (N, K) array of distances
        """
        num_atoms = len(atoms)
        # Use a sufficiently large cutoff to ensure we find at least K neighbors
        # 10.0 Angstroms is generally sufficient for solid state materials
        cutoff = 10.0

        # i: central atom indices, j: neighbor atom indices, d: distances
        i, j, d = neighbor_list("ijd", atoms, cutoff)

        neighbor_indices = np.full((num_atoms, self.k_neighbors), -1, dtype=int)
        neighbor_distances = np.full((num_atoms, self.k_neighbors), np.inf)

        for atom_idx in range(num_atoms):
            # Mask for current atom
            mask = i == atom_idx

            # Get neighbors for this atom
            dists = d[mask]
            inds = j[mask]

            # Sort by distance
            sorted_args = np.argsort(dists)

            # Take top K
            num_found = len(dists)
            num_to_take = min(num_found, self.k_neighbors)

            if num_to_take > 0:
                neighbor_distances[atom_idx, :num_to_take] = dists[sorted_args][
                    :num_to_take
                ]
                neighbor_indices[atom_idx, :num_to_take] = inds[sorted_args][
                    :num_to_take
                ]

        return neighbor_indices, neighbor_distances

    def compute_soft_context(
        self, neighbor_indices, neighbor_distances, atomic_numbers
    ):
        """
        Computes the Normalized Soft-Neighbor Identity vector.
        """
        num_atoms = len(neighbor_indices)
        context_vectors = np.zeros((num_atoms, self.num_atom_types))

        # Map atomic numbers to 0-3 indices
        atom_type_indices = np.array([self.z_to_idx.get(z, -1) for z in atomic_numbers])

        for i in range(num_atoms):
            dists = neighbor_distances[i]
            # Filter out invalid neighbors (infinite distance or index -1)
            valid_mask = (dists < np.inf) & (neighbor_indices[i] != -1)

            if not np.any(valid_mask):
                continue

            # Inverse distance weighting
            # Add small epsilon to avoid division by zero
            weights = 1.0 / (dists[valid_mask] + 1e-6)

            # Normalize weights to sum to 1
            total_weight = np.sum(weights)
            if total_weight > 1e-9:
                weights /= total_weight
            else:
                # Fallback: uniform weights if distances are weirdly uniform/zero
                weights[:] = 1.0 / len(weights)

            # Neighbor types
            nbr_idxs = neighbor_indices[i][valid_mask]
            nbr_types = atom_type_indices[nbr_idxs]

            # Accumulate weighted one-hot vectors
            for w, t_idx in zip(weights, nbr_types):
                if t_idx != -1:
                    context_vectors[i, t_idx] += w

        return context_vectors

    def compute_global_features(self, atoms):
        """
        Extracts macroscopic features: Lattice params, Volume, Density, Stoichiometry.
        """
        # Lattice lengths and angles
        lengths = atoms.cell.lengths()
        angles = atoms.cell.angles()

        # Volume
        volume = atoms.get_volume()

        # Atomic Density
        num_atoms = len(atoms)
        density = num_atoms / volume if volume > 1e-6 else 0.0

        # Stoichiometry (fractions of Al, Ga, In)
        # O is implicitly handled as the remainder or just ignored as it's the anion
        chemical_symbols = atoms.get_chemical_symbols()
        counts = {"Al": 0, "Ga": 0, "In": 0}
        for s in chemical_symbols:
            if s in counts:
                counts[s] += 1

        frac_al = counts["Al"] / num_atoms if num_atoms > 0 else 0
        frac_ga = counts["Ga"] / num_atoms if num_atoms > 0 else 0
        frac_in = counts["In"] / num_atoms if num_atoms > 0 else 0

        # Total Atoms
        total_atoms = float(num_atoms)

        # Construct feature vector (dim 12)
        # 0-2: Lengths
        # 3-5: Angles
        # 6: Volume
        # 7: Density
        # 8-10: Stoichiometry (Al, Ga, In)
        # 11: Total Atoms
        global_feats = np.array(
            [*lengths, *angles, volume, density, frac_al, frac_ga, frac_in, total_atoms]
        )

        return global_feats

    def process_file(self, file_path):
        """
        Main processing function for a single file.
        Returns:
            atomic_features: (N, 12)
            global_features: (12,)
        """
        atoms = self.parse_xyz(file_path)

        # 1. Centered Coordinates
        positions = atoms.get_positions()
        centroid = np.mean(positions, axis=0)
        centered_pos = positions - centroid

        # 2. Neighbors
        nbr_indices, nbr_dists = self.get_k_nearest_neighbors(atoms)

        # 3. d_min (distance to single nearest neighbor)
        # nbr_dists is sorted, so index 0 is closest
        d_min = nbr_dists[:, 0].reshape(-1, 1)

        # 4. Soft Context
        atomic_numbers = atoms.get_atomic_numbers()
        soft_context = self.compute_soft_context(nbr_indices, nbr_dists, atomic_numbers)

        # 5. Atomic Identity One-Hot
        atom_type_indices = [self.z_to_idx.get(z, -1) for z in atomic_numbers]
        one_hot = np.zeros((len(atoms), self.num_atom_types))
        for i, idx in enumerate(atom_type_indices):
            if idx != -1:
                one_hot[i, idx] = 1.0

        # Combine Atomic Features
        # Identity (4) + Coords (3) + d_min (1) + Context (4) = 12
        atomic_features = np.hstack([one_hot, centered_pos, d_min, soft_context])

        # 6. Global Features
        global_features = self.compute_global_features(atoms)

        return atomic_features, global_features


def process_dataset(metadata_path, processor, load_cached_data=True, cache_path=None):
    """
    Processes a dataset defined by a metadata CSV.
    Handles caching logic.
    """
    # Ensure working directory exists
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try loading cache
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "atomic_features": data["atomic_features"],
                "global_features": data["global_features"],
                "batch_indices": data["batch_indices"],
                "targets": data["targets"] if "targets" in data else None,
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    all_atomic_features = []
    all_global_features = []
    batch_indices = [0]  # Start index of each molecule in the flattened atomic array
    ids = []
    targets = []

    print(f"Processing {len(df)} samples from {metadata_path}...")

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            af, gf = processor.process_file(file_path)

            all_atomic_features.append(af)
            all_global_features.append(gf)

            # Track where this molecule ends (cumulative sum of atoms)
            batch_indices.append(batch_indices[-1] + len(af))

            ids.append(row["id"])

            # Targets (if available)
            if "formation_energy_ev_natom" in row and not pd.isna(
                row["formation_energy_ev_natom"]
            ):
                targets.append(
                    [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
                )
            else:
                # Placeholder for test set if needed, though usually we handle test separately
                targets.append([0.0, 0.0])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Concatenate all atomic features into one big array
    if not all_atomic_features:
        raise ValueError("No data processed!")

    stacked_atomic = np.vstack(all_atomic_features).astype(np.float32)
    stacked_global = np.vstack(all_global_features).astype(np.float32)
    batch_indices = np.array(batch_indices, dtype=np.int32)
    ids = np.array(ids, dtype=np.int32)
    targets = np.array(targets, dtype=np.float32)

    # 3. Save to cache
    if cache_path:
        print(f"Saving processed data to {cache_path}")
        np.savez(
            cache_path,
            atomic_features=stacked_atomic,
            global_features=stacked_global,
            batch_indices=batch_indices,
            targets=targets,
            ids=ids,
        )

    return {
        "atomic_features": stacked_atomic,
        "global_features": stacked_global,
        "batch_indices": batch_indices,
        "targets": targets,
        "ids": ids,
    }
