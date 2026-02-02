import numpy as np
from ase import Atoms
from library.config import ATOM_MAP, K_MIN, K_SHORT, K_MEDIUM, K_LONG, NUM_ATOM_TYPES


def compute_pbc_neighbors(atoms: Atoms, k_max: int):
    """
    Computes distances and indices of the k_max nearest neighbors for each atom,
    respecting periodic boundary conditions.

    Args:
        atoms: ASE Atoms object.
        k_max: Number of neighbors to find.

    Returns:
        neighbor_distances: (N, k_max) array of distances.
        neighbor_indices: (N, k_max) array of indices of neighbor atoms in the original list.
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    n_atoms = len(atoms)

    # Generate image offsets (3x3x3 = 27 images)
    ranges = np.arange(-1, 2)
    a_idx, b_idx, c_idx = np.meshgrid(ranges, ranges, ranges, indexing="ij")
    offsets_frac = np.stack([a_idx.flatten(), b_idx.flatten(), c_idx.flatten()], axis=1)
    offsets_cart = offsets_frac @ cell

    # Create supercell positions: (27 * N, 3)
    super_pos = positions[np.newaxis, :, :] + offsets_cart[:, np.newaxis, :]
    super_pos = super_pos.reshape(-1, 3)

    # Map supercell indices back to original indices
    super_indices = np.tile(np.arange(n_atoms), 27)

    # Compute distance matrix between original atoms (N) and supercell atoms (27*N)
    # Shape: (N, 1, 3) - (1, 27*N, 3) -> (N, 27*N, 3)
    diff = positions[:, np.newaxis, :] - super_pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Sort distances to find nearest neighbors
    # We take columns 1 to k_max + 1 because column 0 is the self-interaction (dist=0)
    sorted_indices_all = np.argsort(dists, axis=1)
    neighbor_super_indices = sorted_indices_all[:, 1 : k_max + 1]

    # Retrieve distances and map indices back to original atoms
    neighbor_distances = np.take_along_axis(dists, neighbor_super_indices, axis=1)
    neighbor_indices = super_indices[neighbor_super_indices]

    return neighbor_distances, neighbor_indices


def extract_atomic_features(atoms: Atoms):
    """
    Extracts dense feature vectors for each atom in the structure.

    Features (17 dims):
      - Identity (4): One-hot encoding (Al, Ga, In, O)
      - Spatial Context (3): Centered Cartesian coordinates
      - d_min (1): Distance to nearest neighbor
      - Local Packing Ratio (1): d_min / d_mean_12
      - Context K=6 (4): Weighted composition
      - Context K=24 (4): Weighted composition

    Args:
        atoms: ASE Atoms object.

    Returns:
        features: (N, 17) numpy array.
    """
    n_atoms = len(atoms)
    chemical_symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # 1. Atomic Identity (One-hot)
    identity_features = np.zeros((n_atoms, NUM_ATOM_TYPES))
    for i, symbol in enumerate(chemical_symbols):
        if symbol in ATOM_MAP:
            identity_features[i, ATOM_MAP[symbol]] = 1.0

    # 2. Spatial Context (Centered Coords)
    centroid = np.mean(positions, axis=0)
    centered_coords = positions - centroid

    # 3. Neighbor Calculations
    # Compute neighbors up to K_LONG (24)
    dists, indices = compute_pbc_neighbors(atoms, K_LONG)

    # d_min (K_MIN=1)
    d_min = dists[:, 0:1]  # Shape (N, 1)

    # Local Packing Ratio: d_min / d_mean_12
    d_12 = dists[:, :K_MEDIUM]
    d_mean_12 = np.mean(d_12, axis=1, keepdims=True)

    # Avoid division by zero
    packing_ratio = np.divide(
        d_min, d_mean_12, out=np.zeros_like(d_min), where=d_mean_12 != 0
    )

    # 4. Multi-Scale Chemical Contexts
    def compute_context(k, neighbor_dists, neighbor_idxs):
        # Weights: 1 / (d + epsilon)
        weights = 1.0 / (neighbor_dists + 1e-6)

        context = np.zeros((n_atoms, NUM_ATOM_TYPES))

        for i in range(n_atoms):
            idxs = neighbor_idxs[i]
            w = weights[i]

            for j, neighbor_idx in enumerate(idxs):
                symbol = chemical_symbols[neighbor_idx]
                if symbol in ATOM_MAP:
                    type_idx = ATOM_MAP[symbol]
                    context[i, type_idx] += w[j]

        # Normalize
        row_sums = context.sum(axis=1, keepdims=True)
        context = np.divide(
            context, row_sums, out=np.zeros_like(context), where=row_sums != 0
        )
        return context

    context_6 = compute_context(K_SHORT, dists[:, :K_SHORT], indices[:, :K_SHORT])
    context_24 = compute_context(K_LONG, dists[:, :K_LONG], indices[:, :K_LONG])

    # Concatenate all features
    features = np.hstack(
        [
            identity_features,
            centered_coords,
            d_min,
            packing_ratio,
            context_6,
            context_24,
        ]
    )

    return features.astype(np.float32)


def extract_global_features(atoms: Atoms):
    """
    Extracts global feature vector for the structure.

    Features (15 dims):
      - Lattice Lengths (3)
      - Lattice Angles (3)
      - Volume (1)
      - Density (1)
      - Stoichiometry (3)
      - Total Atoms (1)
      - Lattice Aspect Ratios (3)

    Args:
        atoms: ASE Atoms object.

    Returns:
        features: (15,) numpy array.
    """
    cell = atoms.get_cell()
    cell_lengths = cell.lengths()
    cell_angles = cell.angles()
    volume = cell.volume
    n_atoms = len(atoms)
    density = n_atoms / volume if volume > 0 else 0.0

    # Stoichiometry (Al, Ga, In)
    symbols = atoms.get_chemical_symbols()
    counts = {s: 0 for s in ["Al", "Ga", "In"]}
    for s in symbols:
        if s in counts:
            counts[s] += 1

    stoichiometry = np.array([counts["Al"], counts["Ga"], counts["In"]]) / n_atoms

    # Lattice Aspect Ratios
    a, b, c = cell_lengths
    aspect_ratios = np.array(
        [a / b if b != 0 else 0.0, b / c if c != 0 else 0.0, c / a if a != 0 else 0.0]
    )

    features = np.concatenate(
        [
            cell_lengths,
            cell_angles,
            [volume],
            [density],
            stoichiometry,
            [float(n_atoms)],
            aspect_ratios,
        ]
    )

    return features.astype(np.float32)
