import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import BVS_PARAMS, RDF_MAX_R, RDF_BINS, ANGLE_CUTOFF


def calculate_bond_valences(atoms: Atoms):
    """
    Calculates Bond Valence Sums (Scalar), Vector Bond Valence Sums (Vector),
    and Global Instability Index (GII) for a given structure.

    Args:
        atoms (ase.Atoms): The crystal structure.

    Returns:
        dict: Dictionary containing:
            - 'scalar_bvs': Array of scalar BVS for each atom.
            - 'vector_bvs': Array of vector BVVS magnitudes for each atom.
            - 'gii': Scalar Global Instability Index.
    """
    # Parameters
    b = BVS_PARAMS["b"]

    # Identify atom types
    symbols = np.array(atoms.get_chemical_symbols())
    n_atoms = len(atoms)

    # Initialize arrays
    scalar_bvs = np.zeros(n_atoms)
    vector_bvs = np.zeros((n_atoms, 3))

    # Ideal oxidation states
    ideal_valence = np.zeros(n_atoms)
    for i, sym in enumerate(symbols):
        if sym == "O":
            ideal_valence[i] = 2.0
        elif sym in ["Al", "Ga", "In"]:
            ideal_valence[i] = 3.0

    # Neighbor list for BVS (cutoff 4.0A covers first shell comfortably for these oxides)
    # i: center, j: neighbor, d: distance, D: vector pointing from i to j
    i_indices, j_indices, d_values, D_vectors = neighbor_list("ijdD", atoms, cutoff=4.0)

    for k in range(len(i_indices)):
        idx_i = i_indices[k]
        idx_j = j_indices[k]
        dist = d_values[k]
        vec = D_vectors[k]  # Vector from i to j

        sym_i = symbols[idx_i]
        sym_j = symbols[idx_j]

        # BVS is typically defined for cation-anion pairs
        # Check if one is Metal and other is Oxygen
        is_metal_i = sym_i in ["Al", "Ga", "In"]
        is_oxygen_i = sym_i == "O"
        is_metal_j = sym_j in ["Al", "Ga", "In"]
        is_oxygen_j = sym_j == "O"

        valid_pair = (is_metal_i and is_oxygen_j) or (is_oxygen_i and is_metal_j)

        if valid_pair:
            # Determine R0
            # If i is metal, use its R0. If i is O, use j's R0.
            metal_sym = sym_i if is_metal_i else sym_j
            r0 = BVS_PARAMS.get(metal_sym, 0.0)

            if r0 > 0:
                # Calculate bond valence
                s_ij = np.exp((r0 - dist) / b)

                # Scalar Sum
                scalar_bvs[idx_i] += s_ij

                # Vector Sum (sum of vectors pointing to neighbors weighted by valence)
                # Normalize vector
                if dist > 1e-6:
                    unit_vec = vec / dist
                    vector_bvs[idx_i] += s_ij * unit_vec

    # Calculate Vector Magnitude
    vector_bvs_mag = np.linalg.norm(vector_bvs, axis=1)

    # Calculate GII
    # GII = sqrt( sum( (V_calc - V_ideal)^2 ) / N )
    diff_sq = (scalar_bvs - ideal_valence) ** 2
    gii = np.sqrt(np.mean(diff_sq))

    return {"scalar_bvs": scalar_bvs, "vector_bvs": vector_bvs_mag, "gii": gii}


def calculate_weighted_bond_angles(atoms: Atoms):
    """
    Calculates bond angles weighted by the product of bond valences.
    Separates angles centered on Metals (O-M-O) and Oxygen (M-O-M).

    Args:
        atoms (ase.Atoms): The crystal structure.

    Returns:
        dict: Dictionary containing lists of angles (degrees) and their weights.
    """
    b = BVS_PARAMS["b"]
    symbols = np.array(atoms.get_chemical_symbols())

    # Results containers
    m_centered_angles = []
    m_centered_weights = []
    o_centered_angles = []
    o_centered_weights = []

    # Get neighbors with angle cutoff
    # We need full neighbor list to iterate per atom
    # Using get_neighbors from neighbor_list is efficient for large structures
    # but here we iterate atom by atom to form triplets.

    # Pre-calculate neighbors for all atoms
    # i_indices, j_indices, d_values, D_vectors = neighbor_list('ijdD', atoms, cutoff=ANGLE_CUTOFF)
    # To easily group by center atom, we can use ASE's NeighborList class or just process the arrays

    # Let's process arrays manually for speed
    i_arr, j_arr, d_arr = neighbor_list("ijd", atoms, cutoff=ANGLE_CUTOFF)

    # Pre-compute valences for these bonds to use as weights
    # Note: We calculate valence for ALL pairs within ANGLE_CUTOFF here for weighting
    # This might differ slightly from the BVS function cutoff, but ensures consistency for the angles found.
    valences = np.zeros_like(d_arr)
    for k in range(len(i_arr)):
        idx_i = i_arr[k]
        idx_j = j_arr[k]
        sym_i = symbols[idx_i]
        sym_j = symbols[idx_j]

        is_metal_i = sym_i in ["Al", "Ga", "In"]
        is_oxygen_i = sym_i == "O"
        is_metal_j = sym_j in ["Al", "Ga", "In"]
        is_oxygen_j = sym_j == "O"

        valid_pair = (is_metal_i and is_oxygen_j) or (is_oxygen_i and is_metal_j)

        if valid_pair:
            metal_sym = sym_i if is_metal_i else sym_j
            r0 = BVS_PARAMS.get(metal_sym, 0.0)
            if r0 > 0:
                valences[k] = np.exp((r0 - d_arr[k]) / b)
        else:
            valences[k] = (
                0.0  # Zero weight for non-bonding pairs (e.g. M-M close contacts)
            )

    # Group neighbors by center atom index
    # We assume i_arr is not sorted, so we use an adjacency list approach
    adj = [[] for _ in range(len(atoms))]
    for k, center_idx in enumerate(i_arr):
        neighbor_idx = j_arr[k]
        dist = d_arr[k]
        valence = valences[k]
        # We need the vector for angle calculation.
        # neighbor_list 'D' returns vector pointing from i to j.
        # We re-call neighbor_list with 'D' is expensive? No, we can just call it once.
        # Let's do it cleanly.
        pass

    # Re-doing neighbor call to get vectors efficiently
    i_arr, j_arr, d_arr, D_arr = neighbor_list("ijdD", atoms, cutoff=ANGLE_CUTOFF)

    # Re-map valences (logic is same, just need to match index)
    # Optimization: Calculate valence on the fly inside the loop if needed, but pre-calc is cleaner
    # Let's build the adjacency list with (neighbor_idx, vector, valence)

    # Recalculate valences for this specific list
    bond_valences = []
    for k, dist in enumerate(d_arr):
        idx_i = i_arr[k]
        idx_j = j_arr[k]
        sym_i = symbols[idx_i]
        sym_j = symbols[idx_j]

        is_metal_i = sym_i in ["Al", "Ga", "In"]
        is_oxygen_i = sym_i == "O"
        is_metal_j = sym_j in ["Al", "Ga", "In"]
        is_oxygen_j = sym_j == "O"

        v = 0.0
        if (is_metal_i and is_oxygen_j) or (is_oxygen_i and is_metal_j):
            metal_sym = sym_i if is_metal_i else sym_j
            r0 = BVS_PARAMS.get(metal_sym, 0.0)
            if r0 > 0:
                v = np.exp((r0 - dist) / b)
        bond_valences.append(v)

    # Build adjacency
    neighbors = [[] for _ in range(len(atoms))]
    for k, center_idx in enumerate(i_arr):
        # Store (neighbor_index, vector_from_center_to_neighbor, valence)
        neighbors[center_idx].append((j_arr[k], D_arr[k], bond_valences[k]))

    # Compute Angles
    for i in range(len(atoms)):
        nbs = neighbors[i]
        n_nbs = len(nbs)
        if n_nbs < 2:
            continue

        center_sym = symbols[i]
        is_metal_center = center_sym in ["Al", "Ga", "In"]
        is_oxygen_center = center_sym == "O"

        if not (is_metal_center or is_oxygen_center):
            continue

        # Iterate unique pairs
        for a in range(n_nbs):
            for b in range(a + 1, n_nbs):
                idx_a, vec_a, val_a = nbs[a]
                idx_b, vec_b, val_b = nbs[b]

                # Weight
                w = val_a * val_b

                # Filter low weights to save time/memory?
                # Keep all for now, filter later if needed.
                if w < 1e-6:
                    continue

                # Angle calculation
                # vec_a is vector i->a
                # vec_b is vector i->b
                # angle is between these two
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)

                if norm_a < 1e-6 or norm_b < 1e-6:
                    continue

                dot_prod = np.dot(vec_a, vec_b)
                cosine = dot_prod / (norm_a * norm_b)
                # Clip for numerical stability
                cosine = np.clip(cosine, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cosine))

                if is_metal_center:
                    m_centered_angles.append(angle_deg)
                    m_centered_weights.append(w)
                elif is_oxygen_center:
                    o_centered_angles.append(angle_deg)
                    o_centered_weights.append(w)

    return {
        "M_centered_angles": np.array(m_centered_angles),
        "M_centered_weights": np.array(m_centered_weights),
        "O_centered_angles": np.array(o_centered_angles),
        "O_centered_weights": np.array(o_centered_weights),
    }


def calculate_structural_metrics(atoms: Atoms):
    """
    Calculates Effective Coordination Number (ECoN) and Radial Distribution Functions (RDF).

    Args:
        atoms (ase.Atoms): The crystal structure.

    Returns:
        dict: Dictionary containing:
            - 'econ': Array of ECoN values for each atom.
            - 'rdf': Dictionary of RDF histograms for pairs (e.g., 'Al-O', 'Ga-O').
    """
    # 1. Effective Coordination Number (ECoN)
    # Using a simplified geometric definition: sum of weights based on distance
    # ECoN_i = sum_j exp(1 - (d_ij / d_avg_i)^6)
    # First, we need neighbors. Let's use a generous cutoff for coordination
    cutoff_econ = 4.0
    i_arr, j_arr, d_arr = neighbor_list("ijd", atoms, cutoff=cutoff_econ)

    n_atoms = len(atoms)
    econ = np.zeros(n_atoms)

    # Group distances by atom
    atom_dists = [[] for _ in range(n_atoms)]
    for k, idx in enumerate(i_arr):
        atom_dists[idx].append(d_arr[k])

    for i in range(n_atoms):
        dists = np.array(atom_dists[i])
        if len(dists) == 0:
            econ[i] = 0.0
            continue

        # Calculate weighted average bond length for the shell
        # A common approach for ECoN average distance is an iterative self-consistent one
        # or simply the average of the nearest neighbors.
        # Hoppe's method uses d_av = sum(d * exp(...)) / sum(exp(...)).
        # For simplicity and robustness in ML, we use a fixed weighted average
        # weighting by exp(-d).
        weights = np.exp(-dists)
        if np.sum(weights) > 0:
            d_av = np.average(dists, weights=weights)
        else:
            d_av = np.mean(dists)

        # Calculate ECoN
        # Avoid division by zero
        if d_av > 1e-6:
            terms = np.exp(1.0 - (dists / d_av) ** 6)
            econ[i] = np.sum(terms)
        else:
            econ[i] = 0.0

    # 2. Radial Distribution Functions (RDF)
    # We compute pair-wise distances for specific element pairs
    # Pairs of interest: Al-O, Ga-O, In-O, O-O, Metal-Metal (aggregated)

    symbols = np.array(atoms.get_chemical_symbols())
    rdf_data = {}

    # Define pairs
    pairs_to_compute = [("Al", "O"), ("Ga", "O"), ("In", "O"), ("O", "O")]

    # Get all distances up to RDF_MAX_R
    # neighbor_list is efficient for this
    i_rdf, j_rdf, d_rdf = neighbor_list("ijd", atoms, cutoff=RDF_MAX_R)

    # Create masks for pairs
    sym_i = symbols[i_rdf]
    sym_j = symbols[j_rdf]

    for el1, el2 in pairs_to_compute:
        # Find pairs (i, j) where (sym_i == el1 and sym_j == el2) OR (sym_i == el2 and sym_j == el1)
        # Note: neighbor_list returns both i->j and j->i.
        # For RDF, we want the distribution of distances from type A to type B.
        # If el1 != el2, we look for A->B and B->A?
        # Usually RDF g_AB(r) counts neighbors of type B around A.
        # We will collect all distances for the pair type.

        mask = (sym_i == el1) & (sym_j == el2)

        # If symmetric (e.g. O-O), we avoid double counting if we were iterating unique pairs,
        # but neighbor_list gives directed edges. So for O-O, we get O1-O2 and O2-O1.
        # This is fine for a histogram of "all bond lengths of this type".

        dists = d_rdf[mask]

        # Compute histogram
        hist, bin_edges = np.histogram(
            dists, bins=RDF_BINS, range=(0, RDF_MAX_R), density=False
        )

        # Normalize by number of atoms of type 1 to make it intensive?
        # Or just return density.
        # For ML features, a normalized histogram (density=True) or count / total_volume is good.
        # Let's use density=True to be scale invariant regarding cell size (roughly).
        # Actually, simple count normalization by cell volume is better physically,
        # but density=True makes the shape comparable.

        # However, if an element is missing (e.g. no In), dists is empty.
        if len(dists) > 0:
            hist, _ = np.histogram(
                dists, bins=RDF_BINS, range=(0, RDF_MAX_R), density=True
            )
        else:
            hist = np.zeros(RDF_BINS)

        rdf_data[f"{el1}-{el2}"] = hist

    # Metal-Metal generic RDF
    is_metal_i = np.isin(sym_i, ["Al", "Ga", "In"])
    is_metal_j = np.isin(sym_j, ["Al", "Ga", "In"])
    mask_mm = is_metal_i & is_metal_j
    dists_mm = d_rdf[mask_mm]

    if len(dists_mm) > 0:
        hist_mm, _ = np.histogram(
            dists_mm, bins=RDF_BINS, range=(0, RDF_MAX_R), density=True
        )
    else:
        hist_mm = np.zeros(RDF_BINS)
    rdf_data["Metal-Metal"] = hist_mm

    return {"econ": econ, "rdf": rdf_data}
