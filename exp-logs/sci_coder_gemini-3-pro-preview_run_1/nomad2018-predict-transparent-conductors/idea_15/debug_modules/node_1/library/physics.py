import numpy as np


def get_lattice_matrix(a, b, c, alpha, beta, gamma):
    """
    Converts lattice lengths and angles to a 3x3 lattice matrix representation.
    The vectors are aligned such that v1 is along the x-axis and v2 is in the xy-plane.

    Args:
        a, b, c (float): Lattice vector lengths in Angstroms.
        alpha, beta, gamma (float): Lattice angles in degrees.

    Returns:
        numpy.ndarray: A 3x3 matrix where rows correspond to lattice vectors v1, v2, v3.
    """
    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Vector 1: Aligned with x-axis
    v1 = np.array([a, 0.0, 0.0])

    # Vector 2: In xy-plane
    v2_x = b * np.cos(gamma_rad)
    v2_y = b * np.sin(gamma_rad)
    v2 = np.array([v2_x, v2_y, 0.0])

    # Vector 3: General orientation
    # Derived from dot products: v1.v3 = a*v3x = a*c*cos(beta)
    v3_x = c * np.cos(beta_rad)

    # v2.v3 = v2x*v3x + v2y*v3y = b*c*cos(alpha)
    # Substitute v2x, v3x and solve for v3y
    v3_y = (c * np.cos(alpha_rad) - v3_x * np.cos(gamma_rad)) / np.sin(gamma_rad)

    # v3z is the remaining component to satisfy length c
    # Use max(0, ...) to prevent numerical errors causing sqrt of negative
    v3_z_sq = c**2 - v3_x**2 - v3_y**2
    v3_z = np.sqrt(max(0.0, v3_z_sq))
    v3 = np.array([v3_x, v3_y, v3_z])

    return np.array([v1, v2, v3])


def calculate_cell_volume(lattice_matrix):
    """
    Calculates the volume of the unit cell given the lattice matrix.
    The volume is the absolute value of the determinant of the lattice matrix.

    Args:
        lattice_matrix (numpy.ndarray): 3x3 matrix of lattice vectors.

    Returns:
        float: Volume of the unit cell in Angstroms^3.
    """
    return np.abs(np.linalg.det(lattice_matrix))


def compute_pbc_interactions(coords, lattice_matrix):
    """
    Computes the pairwise distance matrix between atoms considering Periodic Boundary Conditions (PBC)
    using the Minimum Image Convention (MIC).

    Args:
        coords (numpy.ndarray): (N, 3) array of atomic Cartesian coordinates.
        lattice_matrix (numpy.ndarray): (3, 3) array where rows are lattice vectors.

    Returns:
        numpy.ndarray: (N, N) symmetric matrix of pairwise distances.
    """
    # Calculate inverse lattice matrix to convert to fractional coordinates
    # If coords = frac @ lattice, then frac = coords @ inv_lattice
    inv_lattice = np.linalg.inv(lattice_matrix)

    # Convert Cartesian coordinates to fractional coordinates
    frac_coords = coords @ inv_lattice

    # Compute pairwise differences in fractional space
    # Shape: (N, N, 3)
    # frac_diff[i, j, :] = frac_coords[i] - frac_coords[j]
    frac_diff = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention:
    # Shift fractional coordinate differences to be within [-0.5, 0.5]
    frac_diff = frac_diff - np.round(frac_diff)

    # Convert back to Cartesian space
    # cart_diff[i, j, :] = frac_diff[i, j, :] @ lattice_matrix
    cart_diff = frac_diff @ lattice_matrix

    # Compute Euclidean distances
    dist_matrix = np.linalg.norm(cart_diff, axis=-1)

    return dist_matrix


def get_local_potential(dist_matrix, epsilon=1e-12):
    """
    Calculates a scalar Local Potential Proxy for each atom.
    Defined as the sum of inverse distances to all other atoms in the unit cell (nearest images).
    P_i = sum_{j != i} (1 / d_ij)

    Args:
        dist_matrix (numpy.ndarray): (N, N) pairwise distance matrix.
        epsilon (float): Small constant to avoid division by zero if atoms overlap (unlikely)
                         and to handle the self-interaction term.

    Returns:
        numpy.ndarray: (N,) array containing the local potential proxy for each atom.
    """
    # Create a copy to avoid modifying the input matrix
    dists = dist_matrix.copy()

    # Mask the diagonal (self-interaction d_ii = 0) by setting it to infinity
    # This ensures 1/d_ii becomes 0 and doesn't contribute to the sum
    np.fill_diagonal(dists, np.inf)

    # Calculate inverse distances
    # Add epsilon to denominator for numerical stability in case of extremely close atoms
    inv_dists = 1.0 / (dists + epsilon)

    # Sum over neighbors (rows)
    potentials = np.sum(inv_dists, axis=1)

    return potentials


def get_nearest_neighbor(dist_matrix):
    """
    Finds the distance to the nearest neighbor for each atom.

    Args:
        dist_matrix (numpy.ndarray): (N, N) pairwise distance matrix.

    Returns:
        numpy.ndarray: (N,) array containing the distance to the nearest neighbor for each atom.
    """
    # Create a copy
    dists = dist_matrix.copy()

    # Set diagonal to infinity so self-distance (0.0) is not selected as the minimum
    np.fill_diagonal(dists, np.inf)

    # Find the minimum distance along each row
    nn_dists = np.min(dists, axis=1)

    return nn_dists
