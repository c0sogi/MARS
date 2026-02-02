import os
import numpy as np
import pandas as pd
from library.config import CONFIG, PHYSICAL_CONSTANTS, ATOM_MAP
from library.geometry import parse_xyz, calculate_cell_volume, compute_local_fingerprint


def process_dataset(df, root_dir, load_cached_data=True, cache_name="train"):
    """
    Process the dataframe to generate atomic and global features with caching.

    Args:
        df (pd.DataFrame): Dataframe containing metadata.
        root_dir (str): Root directory for input files.
        load_cached_data (bool): Whether to try loading from cache.
        cache_name (str): Name for the cache file (e.g., 'train', 'val', 'test').

    Returns:
        tuple: (atomic_features_list, global_features_array, targets_array, ids_list)
    """
    cache_dir = "./working/idea_6"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}_data.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path)

            all_atomic = data["atomic"]
            counts = data["counts"]
            global_feats = data["global_feats"]
            targets = data["targets"]
            ids = data["ids"]

            # Reconstruct list of atomic features
            atomic_features_list = []
            current_idx = 0
            for count in counts:
                atomic_features_list.append(
                    all_atomic[current_idx : current_idx + count]
                )
                current_idx += count

            return atomic_features_list, global_feats, targets, ids.tolist()
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing {len(df)} samples for {cache_name} set...")

    atomic_features_list = []
    global_features_list = []
    targets_list = []
    ids = []
    atomic_counts = []

    for _, row in df.iterrows():
        # 1. Parse Geometry
        file_path = os.path.join(root_dir, row["file_path"])
        atoms, coords, lattice_vectors = parse_xyz(file_path)

        n_atoms = len(atoms)
        atomic_counts.append(n_atoms)

        # 2. Centered Coordinates
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # 3. Local Fingerprint
        local_stats = compute_local_fingerprint(
            coords, lattice_vectors, k=CONFIG["k_neighbors"]
        )

        # 4. Construct Atomic Features
        sample_atomic_feats = []
        for i in range(n_atoms):
            atom_type = atoms[i]

            # One-hot encoding
            one_hot = [0.0] * 4
            if atom_type in ATOM_MAP:
                one_hot[ATOM_MAP[atom_type]] = 1.0

            # Physical constants
            phys = PHYSICAL_CONSTANTS.get(atom_type, [0.0, 0.0, 0.0])

            # Coords
            xyz = centered_coords[i].tolist()

            # Local stats
            stats = local_stats[i].tolist()

            feat_vec = one_hot + phys + xyz + stats
            sample_atomic_feats.append(feat_vec)

        atomic_features_list.append(np.array(sample_atomic_feats, dtype=np.float32))

        # 5. Construct Global Features
        lv_lengths = [
            row["lattice_vector_1_ang"],
            row["lattice_vector_2_ang"],
            row["lattice_vector_3_ang"],
        ]
        angles = [
            row["lattice_angle_alpha_degree"],
            row["lattice_angle_beta_degree"],
            row["lattice_angle_gamma_degree"],
        ]

        vol = calculate_cell_volume(lattice_vectors)
        # Avoid division by zero if volume is somehow 0
        density = n_atoms / vol if vol > 1e-6 else 0.0

        total_atoms = row["number_of_total_atoms"]
        pct_al = row["percent_atom_al"]
        pct_ga = row["percent_atom_ga"]
        pct_in = row["percent_atom_in"]

        global_vec = (
            lv_lengths
            + angles
            + [vol, density, float(total_atoms), pct_al, pct_ga, pct_in]
        )
        global_features_list.append(np.array(global_vec, dtype=np.float32))

        # 6. Targets
        if "formation_energy_ev_natom" in row and not pd.isna(
            row["formation_energy_ev_natom"]
        ):
            targets_list.append(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            )
        else:
            targets_list.append([0.0, 0.0])

        ids.append(row["id"])

    # Convert to numpy arrays for saving
    # Flatten atomic features for storage
    if atomic_features_list:
        all_atomic_feats = np.vstack(atomic_features_list)
    else:
        all_atomic_feats = np.empty((0, 13))

    global_features_array = np.array(global_features_list, dtype=np.float32)
    targets_array = np.array(targets_list, dtype=np.float32)
    ids_array = np.array(ids)
    counts_array = np.array(atomic_counts, dtype=np.int32)

    # Save to cache
    np.savez(
        cache_path,
        atomic=all_atomic_feats,
        counts=counts_array,
        global_feats=global_features_array,
        targets=targets_array,
        ids=ids_array,
    )
    print(f"Saved processed data to {cache_path}")

    return atomic_features_list, global_features_array, targets_array, ids
