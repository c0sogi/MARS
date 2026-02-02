import numpy as np
import os


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors, atomic species, and coordinates.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
        species (list): List of atomic species strings.
        coords (np.ndarray): Nx3 array of atomic coordinates.
    """
    lattice_vectors = []
    species = []
    coords = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            # Format: atom x y z species
            coords.append([float(x) for x in parts[1:4]])
            species.append(parts[4])

    return np.array(lattice_vectors), species, np.array(coords)


def center_coordinates(coords, lattice_vectors):
    """
    Centers atomic coordinates by subtracting the unit cell centroid.
    The unit cell centroid is defined as 0.5 * (v1 + v2 + v3).

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        centered_coords (np.ndarray): Nx3 array of centered coordinates.
    """
    # Calculate unit cell centroid
    centroid = 0.5 * np.sum(lattice_vectors, axis=0)
    return coords - centroid


def get_pbc_neighbor(coords, species, lattice_vectors):
    """
    Calculates the nearest neighbor distance and identity for each atom,
    considering Periodic Boundary Conditions (PBC).

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        species (list): List of atomic species strings corresponding to coords.
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        nn_distances (np.ndarray): Array of shape (N,) containing distance to nearest neighbor.
        nn_identities (list): List of length N containing species of nearest neighbor.
    """
    n_atoms = len(coords)
    species_arr = np.array(species)

    # Generate image offsets (27 images: -1, 0, 1 for each dimension)
    shifts = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                shifts.append(
                    i * lattice_vectors[0]
                    + j * lattice_vectors[1]
                    + k * lattice_vectors[2]
                )
    shifts = np.array(shifts)  # Shape (27, 3)

    # Create supercell coordinates: (27, N, 3)
    # image_coords[k, j, :] is the position of atom j in shift k
    image_coords = coords[None, :, :] + shifts[:, None, :]

    # Flatten to (27*N, 3) to treat as a single point cloud
    flat_image_coords = image_coords.reshape(-1, 3)
    flat_image_species = np.tile(species_arr, 27)  # Shape (27*N,)

    nn_distances = np.zeros(n_atoms)
    nn_identities = []

    for i in range(n_atoms):
        # Calculate Euclidean distances from atom i to all atoms in the supercell
        dists = np.linalg.norm(flat_image_coords - coords[i], axis=1)

        # Mask out the self-distance (distance approx 0)
        # We use a small epsilon to handle floating point inaccuracies
        mask = dists > 1e-4

        valid_dists = dists[mask]
        valid_species = flat_image_species[mask]

        if len(valid_dists) > 0:
            min_idx = np.argmin(valid_dists)
            nn_distances[i] = valid_dists[min_idx]
            nn_identities.append(valid_species[min_idx])
        else:
            # Fallback if no neighbors found (unlikely in crystal)
            nn_distances[i] = 0.0
            nn_identities.append(species[i])

    return nn_distances, nn_identities


def process_geometry(file_path):
    """
    High-level function to process a geometry file and return features.

    Args:
        file_path (str): Path to geometry.xyz

    Returns:
        dict: containing:
            'coords': Centered coordinates (N, 3)
            'species': List of species (N,)
            'nn_dist': Nearest neighbor distances (N,)
            'nn_species': Nearest neighbor species (N,)
            'lattice': Lattice vectors (3, 3)
    """
    lattice, species, coords = parse_xyz(file_path)

    # Center coordinates relative to unit cell centroid
    centered_coords = center_coordinates(coords, lattice)

    # Get PBC neighbors (distance and identity)
    nn_dist, nn_species = get_pbc_neighbor(coords, species, lattice)

    return {
        "coords": centered_coords,
        "species": species,
        "nn_dist": nn_dist,
        "nn_species": nn_species,
        "lattice": lattice,
    }
