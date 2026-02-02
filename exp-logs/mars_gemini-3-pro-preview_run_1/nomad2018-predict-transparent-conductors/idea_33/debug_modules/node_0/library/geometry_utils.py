import numpy as np
from scipy.spatial.distance import cdist


def parse_xyz(file_path):
    """
    Parses an xyz file to extract lattice vectors, atomic positions, and types.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
        atom_coords (np.ndarray): Nx3 array of atomic coordinates.
        atom_types (list): List of length N containing atomic symbols.
    """
    lattice_vectors = []
    atom_coords = []
    atom_types = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                # Format: atom x y z symbol
                atom_coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    return np.array(lattice_vectors), np.array(atom_coords), atom_types


def get_pbc_distances(coords, lattice_vectors):
    """
    Computes the pairwise distance matrix respecting periodic boundary conditions
    using the minimum image convention.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        dists (np.ndarray): NxN matrix of pairwise distances.
    """
    # Compute inverse lattice for fractional coordinate conversion
    inv_lattice = np.linalg.inv(lattice_vectors)

    # Convert to fractional coordinates
    frac_coords = np.dot(coords, inv_lattice)

    # Compute differences in fractional space: shape (N, N, 3)
    # Broadcasting: (N, 1, 3) - (1, N, 3)
    diff_frac = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention: wrap fractional differences to [-0.5, 0.5]
    diff_frac = diff_frac - np.round(diff_frac)

    # Convert differences back to Cartesian space
    diff_cart = np.dot(diff_frac, lattice_vectors)

    # Compute Euclidean distances
    dists = np.sqrt(np.sum(diff_cart**2, axis=-1))

    return dists


def compute_chemical_densities(
    coords, atom_types, lattice_vectors, all_atom_types, gamma, cutoff
):
    """
    Calculates Gaussian-weighted chemical density fields for each atom.
    Aggregates contributions from all periodic images within the cutoff.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        atom_types (list): List of atomic symbols for each atom.
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
        all_atom_types (list): List of all possible atomic types (e.g. ['Al', 'Ga', 'In', 'O']).
        gamma (float): Width parameter for Gaussian kernel.
        cutoff (float): Cutoff distance for interactions.

    Returns:
        densities (np.ndarray): NxK matrix where K is len(all_atom_types).
    """
    N = len(coords)
    K = len(all_atom_types)
    densities = np.zeros((N, K), dtype=np.float32)

    # Map atom types to indices for fast aggregation
    type_to_idx = {t: i for i, t in enumerate(all_atom_types)}
    atom_type_indices = np.array([type_to_idx[t] for t in atom_types])

    # Determine the number of periodic images needed in each direction
    # We project the cutoff onto the lattice vectors to ensure coverage
    lat_norms = np.linalg.norm(lattice_vectors, axis=1)
    # Ensure at least 1 image if cutoff > 0, otherwise 0
    n_images = np.ceil(cutoff / lat_norms).astype(int)

    # Generate grid of lattice shifts (e.g., -1, 0, 1)
    ranges = [np.arange(-n, n + 1) for n in n_images]
    nx, ny, nz = np.meshgrid(*ranges, indexing="ij")
    # Shape (M, 3) where M is total number of image cells
    shifts_indices = np.stack([nx.flatten(), ny.flatten(), nz.flatten()], axis=1)

    # Iterate over all periodic image shifts
    for shift_idx in shifts_indices:
        # Calculate the Cartesian vector for this lattice shift
        shift_vector = np.dot(shift_idx, lattice_vectors)

        # Shift the coordinates of the "neighbor" atoms (j)
        shifted_coords = coords + shift_vector

        # Compute distances between original atoms (i) and shifted atoms (j)
        # dists[i, j] is distance from atom i (original) to atom j (shifted)
        dists = cdist(coords, shifted_coords)

        # Filter interactions beyond the cutoff
        mask = dists <= cutoff

        # Exclude self-interaction for the central cell (0,0,0 shift)
        if np.all(shift_idx == 0):
            np.fill_diagonal(mask, False)

        # If no atoms are within cutoff for this shift, skip
        if not np.any(mask):
            continue

        # Calculate Gaussian weights
        # exp(-gamma * d^2)
        weights = np.exp(-gamma * (dists**2))
        weights = weights * mask  # Zero out distant interactions

        # Aggregate weights by atomic type
        for k in range(K):
            # Identify columns (neighbors j) that correspond to atom type k
            # atom_type_indices matches the order of coords/shifted_coords
            type_mask = atom_type_indices == k

            if np.any(type_mask):
                # Sum the weights of all neighbors of type k for each atom i
                # weights[:, type_mask] selects columns for type k
                # sum(axis=1) collapses them to a vector of size N
                densities[:, k] += np.sum(weights[:, type_mask], axis=1)

    return densities
