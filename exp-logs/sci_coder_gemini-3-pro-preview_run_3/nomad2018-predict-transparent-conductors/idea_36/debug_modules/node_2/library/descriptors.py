import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import (
    RDF_CUTOFF,
    RDF_BINS,
    BVS_PARAMS,
    BOND_CUTOFF,
    ECON_CUTOFF,
)


def calculate_macroscopic(atoms: Atoms) -> dict:
    """
    Calculates macroscopic properties of the unit cell.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Dictionary containing volume, density, and number of atoms.
    """
    vol = atoms.get_volume()
    masses = atoms.get_masses()
    total_mass = np.sum(masses)
    # Density in AMU / Angstrom^3 (proportional to g/cm^3)
    density = total_mass / vol if vol > 0 else 0.0

    return {"volume": vol, "density": density, "num_atoms": len(atoms)}


def calculate_rdf(atoms: Atoms) -> dict:
    """
    Calculates element-resolved Radial Distribution Functions (RDF).

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Dictionary where keys are element pairs (e.g., 'Al-O') and values are histogram arrays.
    """
    elements = sorted(list(set(atoms.get_chemical_symbols())))
    rdf_data = {}

    # Define bins
    bins = np.linspace(0, RDF_CUTOFF, RDF_BINS + 1)

    # Metals of interest
    metals = ["Al", "Ga", "In"]

    # Calculate neighbor list for RDF cutoff
    # 'd' returns distances
    i_indices, j_indices, distances = neighbor_list("ijd", atoms, RDF_CUTOFF)

    symbols = np.array(atoms.get_chemical_symbols())

    # Helper to compute histogram for a pair of elements
    def compute_pair_hist(elem1, elem2):
        # Filter indices where atom i is elem1 and atom j is elem2
        mask_i = symbols[i_indices] == elem1
        mask_j = symbols[j_indices] == elem2
        mask = mask_i & mask_j

        dists = distances[mask]
        hist, _ = np.histogram(dists, bins=bins, density=False)

        # Normalize by total number of atoms
        norm_hist = hist / len(atoms)
        return norm_hist

    # Metal-Oxygen pairs
    if "O" in elements:
        for m in metals:
            if m in elements:
                hist_mo = compute_pair_hist(m, "O")
                rdf_data[f"{m}-O"] = hist_mo

    # Metal-Metal pairs
    present_metals = [m for m in metals if m in elements]
    for i, m1 in enumerate(present_metals):
        for m2 in present_metals[i:]:
            hist_mm = compute_pair_hist(m1, m2)
            rdf_data[f"{m1}-{m2}"] = hist_mm

    return rdf_data


def calculate_bvs_econ(atoms: Atoms) -> dict:
    """
    Computes Bond Valence Sums (BVS) and Effective Coordination Numbers (ECoN) for each atom.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Dictionary containing 'bvs' and 'econ' arrays (values per atom).
    """
    n_atoms = len(atoms)
    bvs_values = np.zeros(n_atoms)
    econ_values = np.zeros(n_atoms)

    symbols = atoms.get_chemical_symbols()

    # Use the larger cutoff to cover both BVS and ECoN needs
    max_cutoff = max(BOND_CUTOFF, ECON_CUTOFF)
    i_indices, j_indices, distances = neighbor_list("ijd", atoms, max_cutoff)

    # Iterate through neighbor pairs
    for k in range(len(i_indices)):
        idx_i = i_indices[k]
        idx_j = j_indices[k]
        dist = distances[k]
        elem_i = symbols[idx_i]
        elem_j = symbols[idx_j]

        # --- BVS Calculation ---
        # BVS is typically calculated for Metal-Oxygen bonds using specific parameters
        if dist < BOND_CUTOFF:
            is_m_o = elem_i in ["Al", "Ga", "In"] and elem_j == "O"
            is_o_m = elem_i == "O" and elem_j in ["Al", "Ga", "In"]

            if is_m_o:
                # Atom i is Metal
                params = BVS_PARAMS.get(elem_i)
                if params:
                    r0 = params["R0"]
                    b = params["B"]
                    val = np.exp((r0 - dist) / b)
                    bvs_values[idx_i] += val
            elif is_o_m:
                # Atom i is Oxygen, bonded to Metal j
                # Use Metal's parameters for the bond
                params = BVS_PARAMS.get(elem_j)
                if params:
                    r0 = params["R0"]
                    b = params["B"]
                    val = np.exp((r0 - dist) / b)
                    bvs_values[idx_i] += val

        # --- ECoN Calculation ---
        # Using a simple neighbor count within cutoff as a robust proxy for coordination density
        if dist < ECON_CUTOFF:
            econ_values[idx_i] += 1.0

    return {"bvs": bvs_values, "econ": econ_values}


def calculate_angles(atoms: Atoms) -> dict:
    """
    Computes Intra-polyhedral (O-M-O) and Inter-polyhedral (M-O-M) bond angles.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Dictionary containing lists of angles.
              'omo': Dictionary mapping metal element to list of angles.
              'mom': List of M-O-M angles.
    """
    # Get neighbors within BOND_CUTOFF
    # 'D' returns the vector pointing from i to j
    nl = neighbor_list("ijdD", atoms, BOND_CUTOFF)
    i_indices, j_indices, d_vectors, distances = nl

    symbols = atoms.get_chemical_symbols()

    # Organize neighbors by central atom index
    neighbors = {}
    for k, idx_i in enumerate(i_indices):
        if idx_i not in neighbors:
            neighbors[idx_i] = []
        neighbors[idx_i].append(
            {
                "idx_j": j_indices[k],
                "vec": d_vectors[k],
                "dist": distances[k],
                "elem_j": symbols[j_indices[k]],
            }
        )

    omo_angles = {"Al": [], "Ga": [], "In": []}
    mom_angles = []

    for idx_center, neighs in neighbors.items():
        elem_center = symbols[idx_center]

        # Intra-polyhedral O-M-O Angles
        # Center is Metal (M), neighbors are Oxygen (O)
        if elem_center in ["Al", "Ga", "In"]:
            # Filter for Oxygen neighbors
            o_neighs = [n for n in neighs if n["elem_j"] == "O"]

            # Calculate angles between all unique pairs of O neighbors
            for a in range(len(o_neighs)):
                for b in range(a + 1, len(o_neighs)):
                    v1 = o_neighs[a]["vec"]
                    v2 = o_neighs[b]["vec"]

                    # Calculate angle
                    dot = np.dot(v1, v2)
                    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                    # Clip to avoid numerical errors outside [-1, 1]
                    cos_theta = np.clip(dot / norm, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_theta))

                    omo_angles[elem_center].append(angle)

        # Inter-polyhedral M-O-M Angles
        # Center is Oxygen (O), neighbors are Metals (M)
        elif elem_center == "O":
            # Filter for Metal neighbors
            m_neighs = [n for n in neighs if n["elem_j"] in ["Al", "Ga", "In"]]

            # Calculate angles between all unique pairs of M neighbors
            for a in range(len(m_neighs)):
                for b in range(a + 1, len(m_neighs)):
                    v1 = m_neighs[a]["vec"]
                    v2 = m_neighs[b]["vec"]

                    dot = np.dot(v1, v2)
                    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                    cos_theta = np.clip(dot / norm, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_theta))

                    mom_angles.append(angle)

    return {"omo": omo_angles, "mom": mom_angles}
