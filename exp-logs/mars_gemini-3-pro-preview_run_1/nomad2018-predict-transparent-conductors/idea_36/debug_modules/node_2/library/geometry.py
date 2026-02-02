import os
import numpy as np
import pandas as pd
from ase import Atoms
from library.config import Config
from library.utils import cache_data


class GeometryProcessor:
    """
    Handles parsing of geometry files and extraction of geometric features
    for the PG-WDS strategy.
    """

    def __init__(self):
        # Mapping from atomic symbol to integer index for one-hot encoding
        self.atom_type_to_idx = {atype: i for i, atype in enumerate(Config.ATOM_TYPES)}

    def parse_xyz(self, file_path):
        """
        Parses the custom XYZ format provided in the dataset.
        Lines starting with 'lattice_vector' define the unit cell.
        Lines starting with 'atom' define atomic positions and species.

        Args:
            file_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz').

        Returns:
            ase.Atoms: An ASE Atoms object containing positions, symbols, and cell.
        """
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        cell = []
        positions = []
        symbols = []

        with open(full_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == "lattice_vector":
                    # Format: lattice_vector x y z
                    vec = [float(x) for x in parts[1:4]]
                    cell.append(vec)
                elif parts[0] == "atom":
                    # Format: atom x y z Symbol
                    pos = [float(x) for x in parts[1:4]]
                    sym = parts[4]
                    positions.append(pos)
                    symbols.append(sym)

        # Create ASE Atoms object with periodic boundary conditions enabled
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        return atoms

    def compute_pbc_neighbor_distances(self, atoms, k=Config.K_NEIGHBORS):
        """
        Computes distances to the K nearest neighbors for each atom, respecting
        Periodic Boundary Conditions (PBC).

        Uses a 3x3x3 supercell approach to find neighbors, which is robust for
        small unit cells and the required K=12 neighbors.

        Args:
            atoms (ase.Atoms): The atomic structure.
            k (int): Number of nearest neighbors to consider.

        Returns:
            tuple: (d_min, d_mean)
                d_min (np.array): Distance to the single nearest neighbor for each atom.
                d_mean (np.array): Average distance to the K nearest neighbors for each atom.
        """
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        n_atoms = len(atoms)

        # Generate periodic images (3x3x3 supercell offsets)
        # Indices range from -1 to 1 for a, b, c vectors
        ranges = [range(-1, 2) for _ in range(3)]
        # Create meshgrid of indices
        a_idx, b_idx, c_idx = np.meshgrid(*ranges, indexing="ij")
        # Flatten to (27, 3)
        shifts_idx = np.stack(
            [a_idx.flatten(), b_idx.flatten(), c_idx.flatten()], axis=1
        )
        # Convert to Cartesian shifts: (27, 3) @ (3, 3) -> (27, 3)
        shifts_cart = shifts_idx @ cell

        # Vectorized distance calculation
        # P: (N, 3)
        # P_i: (N, 1, 1, 3) -> Source atoms
        # P_j: (1, N, 1, 3) -> Target atoms
        # S_k: (1, 1, 27, 3) -> Shifts

        P = positions
        P_i = P[:, np.newaxis, np.newaxis, :]
        P_j = P[np.newaxis, :, np.newaxis, :]
        S_k = shifts_cart[np.newaxis, np.newaxis, :, :]

        # Vectors from i to (j+shift)
        Vectors = (P_j + S_k) - P_i  # Shape: (N, N, 27, 3)

        # Squared distances
        D2 = np.sum(Vectors**2, axis=-1)  # Shape: (N, N, 27)

        # Flatten the neighbor dimension to find K nearest among all images
        D2_flat = D2.reshape(n_atoms, -1)

        # Take square root to get real distances
        D_flat = np.sqrt(D2_flat)

        # Sort distances for each atom
        D_sorted = np.sort(D_flat, axis=1)

        # The first distance is always 0.0 (self-interaction at shift 0,0,0)
        # We exclude it and take the next K neighbors.
        # Note: If K is larger than available neighbors (unlikely with 27 images),
        # this would need padding, but 27 images * N atoms >> 12.
        nearest_k = D_sorted[:, 1 : k + 1]  # Shape: (N, K)

        # d_min: distance to the single closest neighbor (index 0 of the slice)
        d_min = nearest_k[:, 0]  # Shape: (N,)

        # d_mean: average of K nearest
        d_mean = np.mean(nearest_k, axis=1)  # Shape: (N,)

        return d_min, d_mean

    def extract_geometric_fingerprints(self, atoms):
        """
        Extracts the specific features required for the Atomic Stream of PG-WDS.

        Features per atom:
        1. Species Index (for One-Hot)
        2. Centered Coordinates (x, y, z)
        3. Nearest Neighbor Distance (d_min)
        4. Local Packing Density (d_mean)

        Args:
            atoms (ase.Atoms): The atomic structure.

        Returns:
            dict: Dictionary containing the extracted feature arrays.
        """
        n_atoms = len(atoms)
        if n_atoms == 0:
            return {
                "species_indices": np.array([], dtype=np.int64),
                "centered_coords": np.empty((0, 3), dtype=np.float64),
                "d_min": np.array([], dtype=np.float64),
                "d_mean": np.array([], dtype=np.float64),
            }

        # 1. Atomic Identity
        symbols = atoms.get_chemical_symbols()
        species_indices = np.array(
            [self.atom_type_to_idx[s] for s in symbols], dtype=np.int64
        )

        # 2. Centered Coordinates
        positions = atoms.get_positions()
        # Calculate centroid of the unit cell atoms
        centroid = np.mean(positions, axis=0)
        centered_coords = positions - centroid

        # 3. Geometric Scalars (d_min, d_mean)
        d_min, d_mean = self.compute_pbc_neighbor_distances(atoms, k=Config.K_NEIGHBORS)

        return {
            "species_indices": species_indices,
            "centered_coords": centered_coords,
            "d_min": d_min,
            "d_mean": d_mean,
        }

    def _process_dataset(self, metadata_df):
        """
        Internal function to process a dataframe of materials.
        Iterates through the metadata, parses geometry files, and extracts features.

        Args:
            metadata_df (pd.DataFrame): Dataframe containing 'id' and 'file_path'.

        Returns:
            dict: Dictionary of aggregated data arrays.
        """
        all_species = []
        all_coords = []
        all_d_min = []
        all_d_mean = []

        # Global features lists
        all_lattice_lengths = []
        all_lattice_angles = []
        all_volumes = []
        all_densities = []
        all_stoichiometry = []
        all_num_atoms = []

        ids = []

        print(f"Processing geometry for {len(metadata_df)} samples...")

        for idx, row in metadata_df.iterrows():
            mat_id = row["id"]
            file_path = row["file_path"]

            # Parse geometry
            atoms = self.parse_xyz(file_path)
            n_atoms = len(atoms)

            # --- Atomic Features ---
            feats = self.extract_geometric_fingerprints(atoms)

            all_species.append(feats["species_indices"])
            all_coords.append(feats["centered_coords"])
            all_d_min.append(feats["d_min"])
            all_d_mean.append(feats["d_mean"])

            # --- Global Features ---
            # Lattice parameters
            cell_lengths = atoms.cell.lengths()
            cell_angles = atoms.cell.angles()
            volume = atoms.get_volume()
            density = n_atoms / volume

            # Stoichiometry (Al, Ga, In)
            # Try to get from metadata first for consistency, else calculate
            if "percent_atom_al" in row:
                stoich = np.array(
                    [
                        row["percent_atom_al"],
                        row["percent_atom_ga"],
                        row["percent_atom_in"],
                    ]
                )
            else:
                syms = atoms.get_chemical_symbols()
                counts = {s: syms.count(s) for s in ["Al", "Ga", "In"]}
                total = len(syms)
                # Normalize by total atoms (including O) or just cations?
                # The metadata implies percent of total atoms usually, or relative cation ratio.
                # Let's assume percent of total atoms based on column name.
                stoich = np.array(
                    [
                        counts.get("Al", 0) / total,
                        counts.get("Ga", 0) / total,
                        counts.get("In", 0) / total,
                    ]
                )

            all_lattice_lengths.append(cell_lengths)
            all_lattice_angles.append(cell_angles)
            all_volumes.append(volume)
            all_densities.append(density)
            all_stoichiometry.append(stoich)
            all_num_atoms.append(n_atoms)

            ids.append(mat_id)

        # Organize data into a dictionary
        # Atomic features are ragged (different N per crystal), so we store as object arrays of numpy arrays
        data = {
            "ids": np.array(ids),
            # Atomic features
            "species_indices": np.array(all_species, dtype=object),
            "centered_coords": np.array(all_coords, dtype=object),
            "d_min": np.array(all_d_min, dtype=object),
            "d_mean": np.array(all_d_mean, dtype=object),
            # Global features (Fixed size per crystal)
            "lattice_lengths": np.array(all_lattice_lengths, dtype=np.float32),
            "lattice_angles": np.array(all_lattice_angles, dtype=np.float32),
            "volume": np.array(all_volumes, dtype=np.float32),
            "density": np.array(all_densities, dtype=np.float32),
            "stoichiometry": np.array(all_stoichiometry, dtype=np.float32),
            "num_atoms": np.array(all_num_atoms, dtype=np.float32),
        }

        # Add targets if they exist (Train/Val sets)
        if "formation_energy_ev_natom" in metadata_df.columns:
            targets = metadata_df[
                ["formation_energy_ev_natom", "bandgap_energy_ev"]
            ].values.astype(np.float32)
            data["targets"] = targets

        return data

    def process_data(self, split_name, load_cached_data=True):
        """
        Public method to process a specific dataset split (train, val, or test).
        Utilizes caching to avoid re-computing expensive geometric features.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: The processed data dictionary.
        """
        filename = f"{split_name}_data.npz"

        def _compute():
            # Load metadata CSV
            csv_path = os.path.join(Config.METADATA_DIR, f"{split_name}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Metadata file {csv_path} not found.")

            df = pd.read_csv(csv_path)
            return self._process_dataset(df)

        return cache_data(_compute, filename, load_cached_data=load_cached_data)


def get_geometry_processor():
    """
    Factory function to get an instance of GeometryProcessor.
    """
    return GeometryProcessor()
