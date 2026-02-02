import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import (
    Standardizer,
    load_dipole_moments,
    load_potential_energy,
    load_magnetic_shielding,
    load_mulliken_charges,
)
from library import features


def process_dataset(split_name, load_cached_data=True):
    """
    Processes the dataset for the given split (train, val, test).
    Generates flattened numpy arrays for efficient training.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: A dictionary containing the processed numpy arrays.
    """
    # 1. Define Cache Paths
    cache_dir = Config.CACHE_PATHS.get(split_name)
    if cache_dir is None:
        # Fallback if not explicitly defined in Config
        cache_dir = os.path.join(Config.PROCESSED_DATA_DIR, f"{split_name}_data")

    os.makedirs(cache_dir, exist_ok=True)

    # List of expected files
    expected_files = [
        "node_z.npy",
        "node_pos.npy",
        "node_batch.npy",
        "edge_index.npy",
        "edge_dist.npy",
        "edge_vec.npy",
        "edge_batch.npy",
        "triplet_indices.npy",
        "triplet_angles.npy",
        "target_indices.npy",
        "target_types.npy",
        "target_molecule_indices.npy",
    ]

    if split_name in ["train", "val"]:
        expected_files.extend(
            [
                "target_values.npy",
                "aux_shielding.npy",
                "aux_charges.npy",
                "aux_dipole.npy",
                "aux_potential.npy",
            ]
        )

    # 2. Check Cache
    if load_cached_data:
        all_exist = all(
            os.path.exists(os.path.join(cache_dir, f)) for f in expected_files
        )
        if all_exist:
            print(f"Loading cached {split_name} data from {cache_dir}...")
            data = {}
            for f in expected_files:
                key = f.replace(".npy", "")
                data[key] = np.load(os.path.join(cache_dir, f))
            return data
        else:
            print(f"Cache miss for {split_name}. Processing from scratch...")

    # 3. Load Raw Data
    print(f"Loading raw metadata for {split_name}...")
    if split_name == "train":
        meta_path = Config.TRAIN_META_PATH
    elif split_name == "val":
        meta_path = Config.VAL_META_PATH
    else:
        meta_path = Config.TEST_META_PATH

    df_meta = pd.read_csv(meta_path)

    # Debug Mode: Subsample
    if Config.DEBUG:
        print(f"DEBUG MODE: Subsampling {Config.DEBUG_SAMPLE_SIZE} molecules...")
        unique_mols = df_meta["molecule_name"].unique()
        if len(unique_mols) > Config.DEBUG_SAMPLE_SIZE:
            # Use deterministic sampling
            rng = np.random.RandomState(Config.SEED)
            selected_mols = rng.choice(
                unique_mols, Config.DEBUG_SAMPLE_SIZE, replace=False
            )
            df_meta = df_meta[df_meta["molecule_name"].isin(selected_mols)].reset_index(
                drop=True
            )

    # Load Structures
    print("Loading structures...")
    df_struct = pd.read_csv(Config.STRUCTURES_PATH)
    # Filter structures to only those in the metadata
    relevant_mols = df_meta["molecule_name"].unique()
    df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

    # Load Aux Data (Only for Train/Val usually, but we load if available)
    # Note: Aux data might not cover all molecules in Test, so we handle that.
    aux_dipole = None
    aux_potential = None
    aux_shielding = None
    aux_charges = None

    if split_name in ["train", "val"]:
        print("Loading auxiliary data...")
        df_dip = load_dipole_moments(load_cached_data)
        # Calculate dipole magnitude
        df_dip["dipole_mag"] = np.sqrt(
            df_dip["X"] ** 2 + df_dip["Y"] ** 2 + df_dip["Z"] ** 2
        )
        df_pot = load_potential_energy(load_cached_data)
        df_shield = load_magnetic_shielding(load_cached_data)
        df_charge = load_mulliken_charges(load_cached_data)

        # Index aux data by molecule_name for fast lookup
        df_dip = df_dip.set_index("molecule_name")
        df_pot = df_pot.set_index("molecule_name")

        # Shielding and Charges are per-atom. Group them.
        # We will merge them during molecule iteration or pre-group
        # Optimization: Create a dict of dicts or similar
        # Since df_struct is sorted by mol_name and atom_index, we can align arrays if we sort aux similarly
        # But let's be safe and use lookups or merges.

        # Filter aux to relevant mols
        df_shield = df_shield[df_shield["molecule_name"].isin(relevant_mols)]
        df_charge = df_charge[df_charge["molecule_name"].isin(relevant_mols)]

    # 4. Prepare for Iteration
    # Group structures by molecule
    # df_struct is likely sorted, but let's ensure
    df_struct = df_struct.sort_values(["molecule_name", "atom_index"])
    struct_groups = df_struct.groupby("molecule_name")

    # Group metadata (targets) by molecule
    meta_groups = df_meta.groupby("molecule_name")

    # Group per-atom aux data
    shield_groups = None
    charge_groups = None
    if split_name in ["train", "val"]:
        shield_groups = df_shield.groupby("molecule_name")
        charge_groups = df_charge.groupby("molecule_name")

    # Initialize lists to store flattened data
    list_node_z = []
    list_node_pos = []
    list_node_batch = []

    list_edge_index = []
    list_edge_dist = []
    list_edge_vec = []
    list_edge_batch = []

    list_triplet_indices = []
    list_triplet_angles = []

    list_target_indices = []  # Index into the global edge array
    list_target_types = []
    list_target_values = []
    list_target_mol_indices = []

    list_aux_shielding = []
    list_aux_charges = []
    list_aux_dipole = []
    list_aux_potential = []

    # Counters for offsets
    node_offset = 0
    edge_offset = 0
    mol_idx = 0

    # Type mapping
    type_map = {t: i for i, t in enumerate(Config.COUPLING_TYPES)}

    # 5. Iterate and Process
    print(f"Processing {len(relevant_mols)} molecules...")

    # Iterate over unique molecules in the metadata (guarantees order)
    # We use relevant_mols array to iterate
    for mol_name in relevant_mols:
        # --- Node Features ---
        group = struct_groups.get_group(mol_name)
        atoms = group["atom"].map(Config.ATOM_MAP).values.astype(np.int64)
        pos = group[["x", "y", "z"]].values.astype(np.float32)
        num_atoms = len(atoms)

        list_node_z.append(atoms)
        list_node_pos.append(pos)
        list_node_batch.append(np.full(num_atoms, mol_idx, dtype=np.int64))

        # --- Aux Node Features (Train/Val) ---
        if split_name in ["train", "val"]:
            # Shielding: (num_atoms, 9)
            if mol_name in shield_groups.groups:
                s_group = shield_groups.get_group(mol_name).sort_values("atom_index")
                shield_tensor = s_group[
                    ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
                ].values.astype(np.float32)
            else:
                shield_tensor = np.zeros((num_atoms, 9), dtype=np.float32)
            list_aux_shielding.append(shield_tensor)

            # Charges: (num_atoms,)
            if mol_name in charge_groups.groups:
                c_group = charge_groups.get_group(mol_name).sort_values("atom_index")
                charges = c_group["mulliken_charge"].values.astype(np.float32)
            else:
                charges = np.zeros(num_atoms, dtype=np.float32)
            list_aux_charges.append(charges)

            # Dipole & Potential (Molecule level)
            d_val = 0.0
            if mol_name in df_dip.index:
                d_val = df_dip.loc[mol_name, "dipole_mag"]
            list_aux_dipole.append(d_val)

            p_val = 0.0
            if mol_name in df_pot.index:
                p_val = df_pot.loc[mol_name, "potential_energy"]
            list_aux_potential.append(p_val)

        # --- Graph Construction ---
        # features.process_molecule returns dict with tensors
        geo_data = features.process_molecule(pos, cutoff=Config.CUTOFF)

        edge_index = geo_data["edge_index"].numpy()  # (2, E)
        dist = geo_data["dist"].numpy()
        vec = geo_data["vec"].numpy()
        triplets = geo_data["triplet_indices"].numpy()  # (2, T)
        angles = geo_data["angles"].numpy()

        num_edges = edge_index.shape[1]

        # --- Target Mapping ---
        # Get targets for this molecule
        if mol_name in meta_groups.groups:
            mol_targets = meta_groups.get_group(mol_name)

            # Arrays for this molecule's targets
            t_atom_0 = mol_targets["atom_index_0"].values
            t_atom_1 = mol_targets["atom_index_1"].values
            t_types = mol_targets["type"].map(type_map).values.astype(np.int64)

            if split_name in ["train", "val"]:
                t_values = mol_targets["scalar_coupling_constant"].values.astype(
                    np.float32
                )
                list_target_values.append(t_values)

            list_target_types.append(t_types)
            list_target_mol_indices.append(
                np.full(len(t_types), mol_idx, dtype=np.int64)
            )

            # Map target pairs to edge indices
            # Create a lookup for edges: (u, v) -> local_edge_index
            # Since edges are directed, we look for u->v
            edge_lookup = {}
            for k in range(num_edges):
                u, v = edge_index[0, k], edge_index[1, k]
                edge_lookup[(u, v)] = k

            local_target_indices = []

            # Check each target
            for i in range(len(t_atom_0)):
                u, v = t_atom_0[i], t_atom_1[i]

                # Try to find u->v
                if (u, v) in edge_lookup:
                    idx = edge_lookup[(u, v)]
                # Try v->u (should exist if graph is undirected by distance, but we want consistent direction)
                elif (v, u) in edge_lookup:
                    idx = edge_lookup[(v, u)]
                else:
                    # Target pair not in radius graph (distance > cutoff)
                    # We must add this edge to the graph to allow readout
                    # Calculate geometry for this new edge
                    new_idx = num_edges

                    # Update edge arrays
                    # Add u->v
                    edge_index = np.concatenate([edge_index, [[u], [v]]], axis=1)

                    # Compute dist/vec
                    d_vec = pos[v] - pos[u]
                    d_val = np.linalg.norm(d_vec)
                    dist = np.append(dist, d_val)
                    vec = np.concatenate([vec, [d_vec]], axis=0)

                    # Add to lookup
                    edge_lookup[(u, v)] = new_idx
                    idx = new_idx

                    num_edges += 1
                    # Note: We do NOT update triplets for these "virtual" long-range edges
                    # to save compute/memory, as angles are less relevant for long range.

                # Store the GLOBAL edge index
                local_target_indices.append(idx + edge_offset)

            list_target_indices.append(np.array(local_target_indices, dtype=np.int64))

        # --- Accumulate Graph Data ---
        # Shift indices by node/edge offsets
        list_edge_index.append(edge_index + node_offset)
        list_edge_dist.append(dist)
        list_edge_vec.append(vec)
        list_edge_batch.append(np.full(num_edges, mol_idx, dtype=np.int64))

        if triplets.shape[1] > 0:
            list_triplet_indices.append(triplets + edge_offset)
            list_triplet_angles.append(angles)

        # Update offsets
        node_offset += num_atoms
        edge_offset += num_edges
        mol_idx += 1

    # 6. Concatenate and Save
    print("Concatenating arrays...")
    data = {}

    # Nodes
    data["node_z"] = np.concatenate(list_node_z)
    data["node_pos"] = np.concatenate(list_node_pos)
    data["node_batch"] = np.concatenate(list_node_batch)

    # Edges
    data["edge_index"] = np.concatenate(list_edge_index, axis=1)
    data["edge_dist"] = np.concatenate(list_edge_dist)
    data["edge_vec"] = np.concatenate(list_edge_vec)
    data["edge_batch"] = np.concatenate(list_edge_batch)

    # Triplets
    if list_triplet_indices:
        data["triplet_indices"] = np.concatenate(list_triplet_indices, axis=1)
        data["triplet_angles"] = np.concatenate(list_triplet_angles)
    else:
        data["triplet_indices"] = np.empty((2, 0), dtype=np.int64)
        data["triplet_angles"] = np.empty((0,), dtype=np.float32)

    # Targets
    data["target_indices"] = np.concatenate(list_target_indices)
    data["target_types"] = np.concatenate(list_target_types)
    data["target_molecule_indices"] = np.concatenate(list_target_mol_indices)

    if split_name in ["train", "val"]:
        data["target_values"] = np.concatenate(list_target_values)
        data["aux_shielding"] = np.concatenate(list_aux_shielding)
        data["aux_charges"] = np.concatenate(list_aux_charges)
        data["aux_dipole"] = np.array(list_aux_dipole, dtype=np.float32)
        data["aux_potential"] = np.array(list_aux_potential, dtype=np.float32)

    print(f"Saving {split_name} data to {cache_dir}...")
    for key, val in data.items():
        np.save(os.path.join(cache_dir, f"{key}.npy"), val)

    # 7. Standardization (Train only)
    if split_name == "train":
        print("Computing target statistics...")
        # We need a dataframe for the Standardizer
        # Reconstruct a minimal DF
        df_stats = pd.DataFrame(
            {
                "scalar_coupling_constant": data["target_values"],
                "type": [Config.COUPLING_TYPES[t] for t in data["target_types"]],
            }
        )
        standardizer = Standardizer()
        standardizer.fit(df_stats)
        standardizer.save(Config.CACHE_PATHS["stats"])
        print("Target statistics saved.")

    print(
        f"Processing complete. Nodes: {len(data['node_z'])}, Edges: {data['edge_index'].shape[1]}, Targets: {len(data['target_indices'])}"
    )
    return data
