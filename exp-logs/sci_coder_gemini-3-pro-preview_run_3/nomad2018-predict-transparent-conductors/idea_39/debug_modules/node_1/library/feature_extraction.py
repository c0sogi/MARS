import numpy as np
import pandas as pd
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import Config


def compute_macroscopic_props(atoms: Atoms) -> dict:
    """
    Computes macroscopic properties of the crystal structure.
    """
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    # Avoid division by zero if volume is somehow zero (unlikely for valid crystals)
    density = mass / vol if vol > 1e-5 else 0.0

    return {"vol_per_atom": vol / len(atoms), "density": density}


def compute_elemental_rdfs(atoms: Atoms, cutoff: float, n_bins: int) -> dict:
    """
    Computes element-resolved Radial Distribution Functions (RDFs).
    Focuses on Metal-Metal and Metal-Oxygen pairs.
    """
    elements = ["Al", "Ga", "In", "O"]
    metals = ["Al", "Ga", "In"]

    # Define pairs of interest
    pairs = []
    # Metal-Metal pairs
    for i in range(len(metals)):
        for j in range(i, len(metals)):
            pairs.append(tuple(sorted((metals[i], metals[j]))))
    # Metal-Oxygen pairs
    for m in metals:
        pairs.append(tuple(sorted((m, "O"))))
    # Oxygen-Oxygen pair
    pairs.append(("O", "O"))

    # Calculate all pairwise distances up to cutoff
    # 'i' and 'j' are indices of atoms, 'D' is distance
    i_indices, j_indices, dists = neighbor_list("ijd", atoms, cutoff)

    chemical_symbols = np.array(atoms.get_chemical_symbols())

    features = {}
    bins = np.linspace(0, cutoff, n_bins + 1)

    # Pre-compute indices for each element to speed up masking
    el_indices = {el: np.where(chemical_symbols == el)[0] for el in elements}

    n_atoms = len(atoms)

    for el1, el2 in pairs:
        # Create masks for the pair (el1, el2)
        # We need to capture both (i=el1, j=el2) and (i=el2, j=el1)
        # neighbor_list returns both i-j and j-i, so we can just check symbols[i] and symbols[j]

        # Mask for source atom == el1
        mask_i_el1 = np.isin(i_indices, el_indices[el1])
        # Mask for target atom == el2
        mask_j_el2 = np.isin(j_indices, el_indices[el2])

        # Combined mask for ordered pair
        mask = mask_i_el1 & mask_j_el2

        # Extract distances for this pair type
        d_pair = dists[mask]

        # Compute histogram
        hist, _ = np.histogram(d_pair, bins=bins)

        # Normalize by total number of atoms to make feature intensive
        norm_hist = hist / n_atoms

        prefix = f"rdf_{el1}_{el2}"
        for b_idx, val in enumerate(norm_hist):
            features[f"{prefix}_bin{b_idx}"] = val

    return features


def calculate_local_anisotropy(vectors):
    """
    Calculates the Local Anisotropy Index based on the vector sum of bond directions.
    """
    if len(vectors) == 0:
        return 0.0

    # Normalize bond vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-9] = 1.0
    normalized_vecs = vectors / norms

    # Vector sum of normalized bond vectors
    vec_sum = np.sum(normalized_vecs, axis=0)
    magnitude = np.linalg.norm(vec_sum)

    return magnitude


def calculate_bvs_econ(symbols, i_idx, j_indices_i, dists_i):
    """
    Calculates Bond Valence Sum (BVS) and Effective Coordination Number (ECoN) for a single atom.
    """
    b_param = Config.BVS_B_PARAM
    r0_map = Config.BVS_PARAMS

    center_el = symbols[i_idx]

    bvs = 0.0

    # ECoN (Hoppe-like definition based on nearest neighbor)
    if len(dists_i) == 0:
        return 0.0, 0.0

    d_min = np.min(dists_i)
    if d_min < 1e-3:
        d_min = 1.0  # Safety

    # ECoN calculation using exponential decay relative to closest neighbor
    econ_val = np.sum(np.exp(1.0 - (dists_i / d_min) ** 6))

    # BVS Calculation: Only consider Metal-Oxygen bonds
    # If center is Metal, look for Oxygen neighbors. If center is Oxygen, look for Metal neighbors.
    for j, d in zip(j_indices_i, dists_i):
        neighbor_el = symbols[j]
        pair = sorted([center_el, neighbor_el])

        # Check if it is a Metal-Oxygen pair
        if pair == ["Al", "O"]:
            r0 = r0_map["Al"]
        elif pair == ["Ga", "O"]:
            r0 = r0_map["Ga"]
        elif pair == ["In", "O"]:
            r0 = r0_map["In"]
        else:
            continue  # Skip non M-O bonds for BVS

        term = np.exp((r0 - d) / b_param)
        bvs += term

    return bvs, econ_val


def compute_local_site_descriptors(atoms: Atoms, cutoff: float) -> dict:
    """
    Computes local site descriptors (Anisotropy, BVS, ECoN) and aggregates them by element.
    """
    # Get neighbor list with distance vectors
    # i, j, d, D: source, target, distance, vector(j)-vector(i)
    i_indices, j_indices, dists, vectors = neighbor_list("ijdD", atoms, cutoff)

    symbols = np.array(atoms.get_chemical_symbols())
    unique_elements = ["Al", "Ga", "In", "O"]

    # Dictionary to store lists of values for each element
    data = {el: {"aniso": [], "bvs": [], "econ": []} for el in unique_elements}

    n_atoms = len(atoms)

    # Iterate over each atom to compute its local properties
    for k in range(n_atoms):
        el = symbols[k]

        # Find neighbors of atom k
        mask = i_indices == k

        neigh_vecs = vectors[mask]
        neigh_dists = dists[mask]
        neigh_idxs = j_indices[mask]

        # Calculate descriptors
        aniso = calculate_local_anisotropy(neigh_vecs)
        bvs, econ = calculate_bvs_econ(symbols, k, neigh_idxs, neigh_dists)

        data[el]["aniso"].append(aniso)
        data[el]["bvs"].append(bvs)
        data[el]["econ"].append(econ)

    # Aggregate distributions into features
    features = {}
    percentiles = [0, 25, 50, 75, 100]

    for el in unique_elements:
        for prop in ["aniso", "bvs", "econ"]:
            values = data[el][prop]
            if len(values) > 0:
                vals = np.array(values)
                # Percentiles
                pcts = np.percentile(vals, percentiles)
                for p, v in zip(percentiles, pcts):
                    features[f"local_{el}_{prop}_p{p}"] = v
                # Mean and Std
                features[f"local_{el}_{prop}_mean"] = np.mean(vals)
                features[f"local_{el}_{prop}_std"] = np.std(vals)
            else:
                # Fill with NaN if element not present
                for p in percentiles:
                    features[f"local_{el}_{prop}_p{p}"] = np.nan
                features[f"local_{el}_{prop}_mean"] = np.nan
                features[f"local_{el}_{prop}_std"] = np.nan

    return features


def compute_network_angles(atoms: Atoms, cutoff: float) -> dict:
    """
    Computes bond angle distributions for M-O-M and O-M-O linkages.
    """
    i_indices, j_indices, _, vectors = neighbor_list("ijdD", atoms, cutoff)
    symbols = np.array(atoms.get_chemical_symbols())

    mom_angles = []
    omo_angles = []

    n_atoms = len(atoms)

    for k in range(n_atoms):
        center_el = symbols[k]

        # Get neighbors
        mask = i_indices == k
        neigh_vecs = vectors[mask]
        neigh_idxs = j_indices[mask]
        neigh_syms = symbols[neigh_idxs]

        # M-O-M angles: Center is Oxygen, neighbors are Metals
        if center_el == "O":
            # Filter for metal neighbors
            valid_mask = np.isin(neigh_syms, ["Al", "Ga", "In"])
            valid_vecs = neigh_vecs[valid_mask]

            n_valid = len(valid_vecs)
            if n_valid >= 2:
                # Normalize vectors
                norms = np.linalg.norm(valid_vecs, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                u_vecs = valid_vecs / norms

                # Compute pairwise dot products
                dots = u_vecs @ u_vecs.T
                # Clip to valid domain for arccos
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.degrees(np.arccos(dots))

                # Extract upper triangle (unique pairs, excluding self)
                tri_indices = np.triu_indices(n_valid, k=1)
                mom_angles.extend(angles[tri_indices])

        # O-M-O angles: Center is Metal, neighbors are Oxygen
        elif center_el in ["Al", "Ga", "In"]:
            # Filter for oxygen neighbors
            valid_mask = neigh_syms == "O"
            valid_vecs = neigh_vecs[valid_mask]

            n_valid = len(valid_vecs)
            if n_valid >= 2:
                norms = np.linalg.norm(valid_vecs, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                u_vecs = valid_vecs / norms

                dots = u_vecs @ u_vecs.T
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.degrees(np.arccos(dots))

                tri_indices = np.triu_indices(n_valid, k=1)
                omo_angles.extend(angles[tri_indices])

    # Aggregate angle distributions
    features = {}
    percentiles = [0, 25, 50, 75, 100]

    for name, vals in [("M_O_M", mom_angles), ("O_M_O", omo_angles)]:
        if len(vals) > 0:
            arr = np.array(vals)
            pcts = np.percentile(arr, percentiles)
            for p, v in zip(percentiles, pcts):
                features[f"angle_{name}_p{p}"] = v
            features[f"angle_{name}_mean"] = np.mean(arr)
            features[f"angle_{name}_std"] = np.std(arr)
        else:
            for p in percentiles:
                features[f"angle_{name}_p{p}"] = np.nan
            features[f"angle_{name}_mean"] = np.nan
            features[f"angle_{name}_std"] = np.nan

    return features


def extract_all_features(atoms: Atoms) -> dict:
    """
    Main function to extract all features for a given Atoms object.
    Combines Macroscopic, RDF, Local Site, and Network Topology features.
    """
    features = {}

    # 1. Macroscopic Properties
    features.update(compute_macroscopic_props(atoms))

    # 2. Radial Distribution Functions
    features.update(
        compute_elemental_rdfs(
            atoms, cutoff=Config.RDF_CUTOFF, n_bins=Config.RDF_NUM_BINS
        )
    )

    # 3. Local Site Descriptors
    features.update(compute_local_site_descriptors(atoms, cutoff=Config.BOND_CUTOFF))

    # 4. Network Topology (Angles)
    features.update(compute_network_angles(atoms, cutoff=Config.BOND_CUTOFF))

    return features
