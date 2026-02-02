import numpy as np


def calculate_pbc_distances(coords, lattice):
    """
    Computes pairwise distances between atoms respecting periodic boundary conditions
    using the Minimum Image Convention (MIC).

    Args:
        coords (np.ndarray): (N, 3) array of atomic coordinates.
        lattice (np.ndarray): (3, 3) array representing the lattice vectors.

    Returns:
        np.ndarray: (N, N) matrix of pairwise distances.
    """
    # Ensure inputs are numpy arrays
    coords = np.array(coords)
    lattice = np.array(lattice)

    # Compute fractional coordinates
    # r = f * L  =>  f = r * L^-1
    # We use solve or inv. Inv is fine for 3x3.
    inv_lattice = np.linalg.inv(lattice)
    frac_coords = np.dot(coords, inv_lattice)

    # Compute differences in fractional coordinates
    # Shape: (N, N, 3)
    # diff[i, j, :] = frac_coords[i] - frac_coords[j]
    # Using broadcasting to create the difference matrix
    diff = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention
    # Wrap fractional differences to the range [-0.5, 0.5]
    diff = diff - np.round(diff)

    # Convert back to Cartesian coordinates
    # cart_diff = diff * L
    # We need to dot the last dimension (3) with the lattice
    cart_diff = np.dot(diff, lattice)

    # Compute Euclidean distances
    distances = np.sqrt(np.sum(cart_diff**2, axis=-1))

    return distances


def get_unit_cell_volume(lengths, angles):
    """
    Calculates the volume of a unit cell given lattice lengths and angles.

    Args:
        lengths (list or np.ndarray): [a, b, c] lattice vector lengths.
        angles (list or np.ndarray): [alpha, beta, gamma] lattice angles in degrees.

    Returns:
        float: The volume of the unit cell.
    """
    a, b, c = lengths
    alpha, beta, gamma = np.radians(angles)

    # Volume formula for a general triclinic cell
    term = (
        1
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )

    # Ensure non-negative value inside sqrt for numerical stability
    volume = a * b * c * np.sqrt(max(0, term))

    return volume


def get_atomic_density(n_atoms, volume):
    """
    Calculates the atomic density of the material.

    Args:
        n_atoms (int): Total number of atoms in the unit cell.
        volume (float): Volume of the unit cell.

    Returns:
        float: Atomic density (atoms per unit volume).
    """
    if volume == 0:
        return 0.0
    return n_atoms / volume
