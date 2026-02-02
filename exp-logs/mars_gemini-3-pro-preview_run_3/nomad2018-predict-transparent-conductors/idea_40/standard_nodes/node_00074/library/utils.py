import numpy as np

# ==========================================
# Scientific Constants
# ==========================================

# Nominal oxidation states for elements in the dataset
OXIDATION_STATES = {"Al": 3, "Ga": 3, "In": 3, "O": -2}

# Bond Valence Parameters (R0)
# Source: Brown, I. D. & Altermatt, D. (1985). Acta Cryst. B41, 244-247.
# Keys are tuples of (Cation, Anion).
# The B parameter is universally taken as 0.37 Angstroms.
BVS_PARAMS = {
    ("Al", "O"): 1.651,
    ("Ga", "O"): 1.730,
    ("In", "O"): 1.907,
    # Reverse mapping for convenience
    ("O", "Al"): 1.651,
    ("O", "Ga"): 1.730,
    ("O", "In"): 1.907,
}

BVS_B = 0.37

# Shannon Ionic Radii (Angstroms) for typical coordination (e.g., VI for cations, IV for O)
# These are approximate values useful for geometric heuristics.
IONIC_RADII = {"Al": 0.54, "Ga": 0.62, "In": 0.80, "O": 1.40}

# ==========================================
# Mathematical & Geometric Utilities
# ==========================================


def compute_percentiles(values, percentiles=[0, 25, 50, 75, 100]):
    """
    Computes specified percentiles of a distribution.

    Args:
        values (list or np.array): Input numerical data.
        percentiles (list): List of percentiles to compute (0-100).

    Returns:
        np.array: Calculated percentiles. Returns zeros if input is empty.
    """
    values = np.array(values)
    # Remove NaNs
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.zeros(len(percentiles))

    return np.percentile(values, percentiles)


def calculate_bvs(element_symbol, neighbor_symbols, distances):
    """
    Calculates the Bond Valence Sum (BVS) for a central atom.

    BVS = sum( exp( (R0 - d_i) / B ) )

    Args:
        element_symbol (str): Symbol of the central atom.
        neighbor_symbols (list of str): Symbols of neighboring atoms.
        distances (list of float): Distances to neighboring atoms (Angstroms).

    Returns:
        float: The calculated Bond Valence Sum.
    """
    bvs = 0.0
    for n_sym, dist in zip(neighbor_symbols, distances):
        # We check both orders in the dictionary
        key = (element_symbol, n_sym)
        if key in BVS_PARAMS:
            r0 = BVS_PARAMS[key]
            # BVS formula
            val = np.exp((r0 - dist) / BVS_B)
            bvs += val
    return bvs


def calculate_econ(distances):
    """
    Calculates the Effective Coordination Number (ECoN) using the exponential weighting method.

    ECoN = sum_i exp( 1 - (d_i / d_min)^6 )

    Args:
        distances (list of float): Distances to neighboring atoms.

    Returns:
        float: The effective coordination number.
    """
    if len(distances) == 0:
        return 0.0

    d_array = np.array(distances)
    # Filter out very small distances (self-interaction or errors)
    d_array = d_array[d_array > 1e-3]

    if len(d_array) == 0:
        return 0.0

    d_min = np.min(d_array)

    # ECoN formula: sum( exp(1 - (d_i/d_min)^6) )
    # This weights neighbors based on how close they are relative to the closest neighbor.
    weights = np.exp(1.0 - (d_array / d_min) ** 6)
    return np.sum(weights)


def get_pbc_displacement(pos1, pos2, cell, pbc=True):
    """
    Calculates the displacement vector from pos1 to pos2 under periodic boundary conditions.
    Using the minimum image convention (MIC) via fractional coordinates.

    Args:
        pos1 (np.array): Coordinate of point 1.
        pos2 (np.array): Coordinate of point 2.
        cell (np.array): 3x3 Lattice matrix (row vectors).
        pbc (bool): Whether to apply PBC.

    Returns:
        np.array: Displacement vector (pos2 - pos1) in Cartesian coordinates.
        float: Distance (norm of displacement).
    """
    diff = pos2 - pos1
    if not pbc:
        return diff, np.linalg.norm(diff)

    # Convert to fractional coordinates
    # cell is row-major: v_cart = v_frac * cell
    # v_frac = v_cart * inv(cell)
    try:
        inv_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        # Fallback for singular cell (should not happen in valid crystals)
        return diff, np.linalg.norm(diff)

    diff_frac = np.dot(diff, inv_cell)

    # Apply MIC: shift fractional coordinates to [-0.5, 0.5]
    diff_frac -= np.round(diff_frac)

    # Convert back to Cartesian
    diff_cart = np.dot(diff_frac, cell)
    dist = np.linalg.norm(diff_cart)

    return diff_cart, dist


def calculate_local_anisotropy(displacements):
    """
    Calculates a simple measure of local geometric anisotropy based on neighbor vectors.

    Anisotropy = | sum( v_i / |v_i| ) | / N

    Args:
        displacements (list of np.array): List of displacement vectors to neighbors.

    Returns:
        float: Anisotropy index (0 = perfectly symmetric/isotropic, 1 = all neighbors in one direction).
    """
    if len(displacements) == 0:
        return 0.0

    sum_vec = np.zeros(3)
    count = 0
    for vec in displacements:
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            sum_vec += vec / norm
            count += 1

    if count == 0:
        return 0.0

    return np.linalg.norm(sum_vec) / count
