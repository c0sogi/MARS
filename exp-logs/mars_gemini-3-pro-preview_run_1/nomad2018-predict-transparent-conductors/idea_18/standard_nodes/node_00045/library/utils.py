import numpy as np
from library.config import Config


def parse_xyz(file_path):
    """
    Parses the custom XYZ format provided in the dataset.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        dict: A dictionary containing:
            - 'lattice_vectors': np.ndarray of shape (3, 3)
            - 'atom_types': np.ndarray of shape (N,)
            - 'coords': np.ndarray of shape (N, 3)
    """
    lattice_vectors = []
    atom_types = []
    coords = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                # Format: atom x y z type
                coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    coords_arr = np.array(coords, dtype=np.float32)
    # Cite debug_lesson_4: Enforce Feature Dimensions on Empty Arrays
    if coords_arr.size == 0:
        coords_arr = coords_arr.reshape(0, 3)

    return {
        "lattice_vectors": np.array(lattice_vectors, dtype=np.float32),
        "atom_types": np.array(atom_types),
        "coords": coords_arr,
    }


def calculate_cell_volume(lattice_vectors):
    """
    Calculates the volume of the unit cell using the determinant of the lattice matrix.

    Args:
        lattice_vectors (np.ndarray): Shape (3, 3).

    Returns:
        float: Volume in Angstrom^3.
    """
    return np.abs(np.linalg.det(lattice_vectors))


def calculate_atomic_density(num_atoms, volume):
    """
    Calculates the atomic density of the crystal.

    Args:
        num_atoms (int): Total number of atoms.
        volume (float): Unit cell volume.

    Returns:
        float: Density in atoms/Angstrom^3.
    """
    if volume == 0:
        return 0.0
    return num_atoms / volume


def get_pbc_distances(coords, lattice_vectors):
    """
    Computes the pairwise distance matrix respecting Periodic Boundary Conditions (PBC)
    using the Minimum Image Convention (MIC) on fractional coordinates.

    Args:
        coords (np.ndarray): Shape (N, 3), Cartesian coordinates.
        lattice_vectors (np.ndarray): Shape (3, 3).

    Returns:
        np.ndarray: Shape (N, N), pairwise distances.
    """
    # Transform coordinates to fractional space: r_frac = r_cart * L^-1
    # Note: We use the inverse of the lattice matrix.
    inv_lattice = np.linalg.inv(lattice_vectors)
    frac_coords = np.dot(coords, inv_lattice)

    # Compute pairwise differences in fractional space
    # Shape (N, N, 3) via broadcasting
    delta_frac = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention: wrap fractional differences to [-0.5, 0.5]
    delta_frac = delta_frac - np.round(delta_frac)

    # Transform differences back to Cartesian space: d_cart = d_frac * L
    delta_cart = np.dot(delta_frac, lattice_vectors)

    # Compute Euclidean distances
    distances = np.sqrt(np.sum(delta_cart**2, axis=2))

    return distances


def get_chemical_neighbor_distances(
    coords, atom_types, lattice_vectors, target_types=None
):
    """
    Generates a vector of distances to the nearest atom of each specified type for every atom.

    For each atom i and each target element type T:
    1. Identify all atoms j of type T.
    2. Calculate PBC distances between i and all j.
    3. If i is of type T, exclude the self-distance (0).
    4. Find the minimum distance.
    5. If no atoms of type T exist, use Config.MAX_NEIGHBOR_DISTANCE.

    Args:
        coords (np.ndarray): Shape (N, 3).
        atom_types (np.ndarray): Shape (N,), string types of each atom.
        lattice_vectors (np.ndarray): Shape (3, 3).
        target_types (list): List of atom types to look for. Defaults to Config.ATOM_TYPES.

    Returns:
        np.ndarray: Shape (N, len(target_types)). Each row corresponds to an atom,
                    containing distances to the nearest neighbor of each target type.
    """
    if target_types is None:
        target_types = Config.ATOM_TYPES

    num_atoms = len(atom_types)
    dist_matrix = get_pbc_distances(coords, lattice_vectors)

    # Initialize result array with MAX_NEIGHBOR_DISTANCE
    chemical_distances = np.full(
        (num_atoms, len(target_types)), Config.MAX_NEIGHBOR_DISTANCE, dtype=np.float32
    )

    # Calculate min self-image distance (approximate as min lattice vector length)
    # This acts as the distance to "self" in the next periodic cell if no other atoms of same type exist.
    lattice_norms = np.linalg.norm(lattice_vectors, axis=1)
    min_lattice_dist = np.min(lattice_norms)

    for t_idx, target_type in enumerate(target_types):
        # Indices of atoms that are of the target type
        target_indices = np.where(atom_types == target_type)[0]

        if len(target_indices) == 0:
            # If no atoms of this type exist, keep MAX_NEIGHBOR_DISTANCE
            continue

        # Extract columns from distance matrix corresponding to the target type atoms
        # Shape (N, num_targets_present)
        dists_to_targets = dist_matrix[:, target_indices]

        # Mask self-distances (where distance is 0 because i == j)
        # We create a boolean mask of shape (N, num_targets_present)
        mask = np.zeros_like(dists_to_targets, dtype=bool)

        # target_indices maps: column_index_k -> atom_index_j
        for col_k, atom_j in enumerate(target_indices):
            # The row corresponding to atom_j in the distance matrix represents
            # the distances FROM atom_j. In the extracted sub-matrix 'dists_to_targets',
            # the element at [atom_j, col_k] is the distance from atom_j to itself.
            mask[atom_j, col_k] = True

        # Apply mask: set self-distances to infinity so they aren't picked as min
        dists_to_targets_masked = dists_to_targets.copy()
        dists_to_targets_masked[mask] = np.inf

        # Find minimum distance for each atom to any atom of target_type
        min_dists = np.min(dists_to_targets_masked, axis=1)

        # Handle cases where the only atom of type T was self (min_dists is inf)
        # In this case, the nearest neighbor is the periodic image of self.
        min_dists[np.isinf(min_dists)] = min_lattice_dist

        chemical_distances[:, t_idx] = min_dists

    # Clip distances to avoid extreme values affecting the model
    chemical_distances = np.minimum(chemical_distances, Config.MAX_NEIGHBOR_DISTANCE)

    return chemical_distances
