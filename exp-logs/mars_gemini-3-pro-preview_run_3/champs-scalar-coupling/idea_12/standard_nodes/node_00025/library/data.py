import os
import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor
from library.config import Config


def process_molecule_task(args):
    """
    Worker function to process a single molecule into graph arrays.
    Calculates nodes, radius graph edges, triplets (for angles), and coupling targets.
    """
    mol_name, struct_df, meta_df, aux_data = args

    # --- 1. Nodes ---
    if struct_df is None:
        # Cite debug_lesson_14: Maintain Referential Integrity When Imputing Missing Graph Data
        # Ensure dummy molecule has enough atoms to satisfy indices in meta_df
        max_idx = 0
        if meta_df is not None and not meta_df.empty:
            max_idx = max(meta_df["atom_index_0"].max(), meta_df["atom_index_1"].max())

        num_nodes = int(max_idx) + 1
        # Create dummy atoms (Carbon=1) and random coords
        atoms = np.ones(num_nodes, dtype=np.int64) * Config.ATOM_MAP.get("C", 1)
        coords = np.random.rand(num_nodes, 3).astype(np.float32)
    else:
        # Map atom symbols to integers
        atoms = struct_df["atom"].map(Config.ATOM_MAP).values.astype(np.int64)
        coords = struct_df[["x", "y", "z"]].values.astype(np.float32)
        num_nodes = len(atoms)

    # --- 2. Edges (Radius Graph) ---
    # Use KDTree for efficient neighbor search within cutoff
    tree = KDTree(coords)
    pairs = tree.query_pairs(Config.CUTOFF)

    u_list, v_list, dist_list = [], [], []
    for i, j in pairs:
        d = np.linalg.norm(coords[i] - coords[j])
        # Add bidirectional edges
        u_list.extend([i, j])
        v_list.extend([j, i])
        dist_list.extend([d, d])

    edge_index = np.array([u_list, v_list], dtype=np.int64)
    edge_dist = np.array(dist_list, dtype=np.float32)
    num_edges = len(dist_list)

    # Create a lookup for (u, v) -> edge_index to map couplings later
    edge_lookup = {(u, v): idx for idx, (u, v) in enumerate(zip(u_list, v_list))}

    # --- 3. Triplets (k -> j -> i) ---
    # Needed for angular features. We find pairs of edges sharing a central node j.
    # Build adjacency list
    incoming = {n: [] for n in range(num_nodes)}
    outgoing = {n: [] for n in range(num_nodes)}
    for idx, (u, v) in enumerate(zip(u_list, v_list)):
        incoming[v].append(idx)
        outgoing[u].append(idx)

    trip_src, trip_dst, trip_angle = [], [], []
    for j in range(num_nodes):
        # For every incoming edge k->j
        for e_in in incoming[j]:
            k = u_list[e_in]
            vec_kj = coords[j] - coords[k]

            # For every outgoing edge j->i
            for e_out in outgoing[j]:
                i = v_list[e_out]
                if k == i:
                    continue  # Skip backtracking

                vec_ji = coords[i] - coords[j]

                # Calculate Cosine Angle
                denom = edge_dist[e_in] * edge_dist[e_out] + 1e-9
                cos = np.dot(vec_kj, vec_ji) / denom

                trip_src.append(e_in)
                trip_dst.append(e_out)
                trip_angle.append(np.clip(cos, -1.0, 1.0))

    triplet_index = np.array([trip_src, trip_dst], dtype=np.int64)
    triplet_angle = np.array(trip_angle, dtype=np.float32)

    # --- 4. Couplings (Targets) ---
    c_types, c_values, c_node_indices, c_edge_indices = [], [], [], []
    if meta_df is not None and not meta_df.empty:
        c_node_indices = meta_df[["atom_index_0", "atom_index_1"]].values.astype(
            np.int64
        )
        c_types = [Config.COUPLING_MAP[t] for t in meta_df["type"]]
        c_values = meta_df.get(
            "scalar_coupling_constant", pd.Series([0] * len(meta_df))
        ).values.astype(np.float32)

        # Map atom pairs to the specific edge index connecting them
        for u, v in c_node_indices:
            c_edge_indices.append(edge_lookup.get((u, v), 0))

    c_types = np.array(c_types, dtype=np.int64)
    c_edge_indices = np.array(c_edge_indices, dtype=np.int64)

    # --- 5. Auxiliary Data ---
    aux_s = None
    if "shielding" in aux_data:
        aux_s = aux_data["shielding"][
            ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
        ].values.astype(np.float32)

    aux_c = None
    if "charges" in aux_data:
        aux_c = (
            aux_data["charges"]["mulliken_charge"]
            .values.astype(np.float32)
            .reshape(-1, 1)
        )

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_triplets": len(trip_angle),
        "num_couplings": len(c_types),
        "atom_types": atoms,
        "edge_index": edge_index,
        "edge_dist": edge_dist,
        "triplet_index": triplet_index,
        "triplet_angle": triplet_angle,
        "coupling_node_indices": c_node_indices,
        "coupling_edge_indices": c_edge_indices,
        "coupling_types": c_types,
        "coupling_values": c_values,
        "aux_shielding": aux_s,
        "aux_charges": aux_c,
    }


def preprocess_data(load_cached_data=True):
    """
    Main data processing function.
    Parses raw files, computes graph features in parallel, and saves flattened SoA arrays.
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "processed")
    os.makedirs(cache_dir, exist_ok=True)

    # Check cache
    if load_cached_data and os.path.exists(
        os.path.join(cache_dir, "train_mol_map.npy")
    ):
        print("Loading cached processed data...")
        return cache_dir

    print("Processing data from scratch...")

    # Load Raw Data
    struct_df = pd.read_csv(Config.STRUCTURES_CSV)
    shielding = pd.read_csv(Config.MAGNETIC_SHIELDING_PATH)
    charges = pd.read_csv(Config.MULLIKEN_CHARGES_PATH)

    # Group for fast access
    struct_grp = struct_df.groupby("molecule_name")
    shield_grp = shielding.groupby("molecule_name")
    charge_grp = charges.groupby("molecule_name")

    splits = [
        ("train", Config.TRAIN_META_PATH),
        ("val", Config.VAL_META_PATH),
        ("test", Config.TEST_META_PATH),
    ]

    for split_name, path in splits:
        meta = pd.read_csv(path)
        meta_grp = meta.groupby("molecule_name")
        mol_names = sorted(meta_grp.groups.keys())  # Deterministic order

        # Prepare tasks
        tasks = []
        for m in mol_names:
            # Cite debug_lesson_2: Verify integrity before access
            if m not in struct_grp.groups:
                if split_name == "test":
                    # Cite debug_lesson_3: Do not drop test data. Use imputation.
                    print(
                        f"Warning: Molecule {m} not found in structures. Imputing dummy structure for Test."
                    )
                    tasks.append((m, None, meta_grp.get_group(m), {}))
                else:
                    print(f"Warning: Molecule {m} not found in structures. Skipping.")
                continue

            aux = {}
            if split_name != "test":
                if m in shield_grp.groups:
                    aux["shielding"] = shield_grp.get_group(m)
                if m in charge_grp.groups:
                    aux["charges"] = charge_grp.get_group(m)
            tasks.append((m, struct_grp.get_group(m), meta_grp.get_group(m), aux))

        print(f"Processing {split_name} ({len(tasks)} molecules)...")

        # Parallel Execution
        results = []
        with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as pool:
            for res in pool.map(process_molecule_task, tasks):
                results.append(res)

        if not results:
            print(f"Warning: No valid molecules found for split {split_name}.")
            # Save empty map to prevent crashes in SOADataset
            mol_map = np.zeros((0, 8), dtype=np.int64)
            np.save(os.path.join(cache_dir, f"{split_name}_mol_map.npy"), mol_map)
            continue

        # Aggregate into Flat Arrays
        arrays = {k: [] for k in results[0].keys() if not k.startswith("num_")}
        mol_map = []

        # Pointers for global indexing
        ptrs = {"n": 0, "e": 0, "t": 0, "c": 0}

        for r in results:
            # Store start index and length for each feature type
            mol_map.append(
                [
                    ptrs["n"],
                    r["num_nodes"],
                    ptrs["e"],
                    r["num_edges"],
                    ptrs["t"],
                    r["num_triplets"],
                    ptrs["c"],
                    r["num_couplings"],
                ]
            )
            ptrs["n"] += r["num_nodes"]
            ptrs["e"] += r["num_edges"]
            ptrs["t"] += r["num_triplets"]
            ptrs["c"] += r["num_couplings"]

            for k in arrays:
                if r[k] is not None:
                    arrays[k].append(r[k])

        # Save Map
        np.save(
            os.path.join(cache_dir, f"{split_name}_mol_map.npy"),
            np.array(mol_map, dtype=np.int64),
        )

        # Save Arrays
        for k, v in arrays.items():
            if v:
                # Determine concatenation axis
                axis = 1 if k in ["edge_index", "triplet_index"] else 0
                if k == "coupling_node_indices":
                    axis = 0

                if len(v) > 0:
                    # Handle empty arrays in v gracefully
                    try:
                        concatenated = np.concatenate(v, axis=axis)
                        np.save(
                            os.path.join(cache_dir, f"{split_name}_{k}.npy"),
                            concatenated,
                        )
                    except ValueError:
                        print(f"Warning: Could not concatenate {k} for {split_name}")

    return cache_dir


class SOADataset:
    """
    Container for the Flattened Structure-of-Arrays data.
    Loads the monolithic numpy arrays into memory.
    """

    def __init__(self, split, data_dir):
        self.split = split
        self.data_dir = data_dir

        # Load molecule map: [n_start, n_len, e_start, e_len, t_start, t_len, c_start, c_len]
        map_path = os.path.join(data_dir, f"{split}_mol_map.npy")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Data not found for split {split} at {data_dir}")

        self.mol_map = np.load(map_path)
        self.num_mols = len(self.mol_map)

        # Load data arrays
        self.data = {}
        keys = [
            "atom_types",
            "edge_index",
            "edge_dist",
            "triplet_index",
            "triplet_angle",
            "coupling_node_indices",
            "coupling_edge_indices",
            "coupling_types",
            "coupling_values",
            "aux_shielding",
            "aux_charges",
        ]

        for k in keys:
            path = os.path.join(data_dir, f"{split}_{k}.npy")
            if os.path.exists(path):
                self.data[k] = np.load(path)
            else:
                self.data[k] = None


class SOALoader:
    """
    Custom DataLoader that iterates over the SOADataset.
    Efficiently slices batches from the monolithic arrays and moves them to the GPU.
    """

    def __init__(self, dataset, batch_size, shuffle=False, device="cpu"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.num_mols = dataset.num_mols

    def __len__(self):
        return (self.num_mols + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.num_mols)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_mols, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_mols)
            batch_indices = indices[start_idx:end_idx]
            yield self._collate(batch_indices)

    def _collate(self, batch_indices):
        # Extract metadata for the batch
        rows = self.dataset.mol_map[batch_indices]
        n_start, n_len = rows[:, 0], rows[:, 1]
        e_start, e_len = rows[:, 2], rows[:, 3]
        t_start, t_len = rows[:, 4], rows[:, 5]
        c_start, c_len = rows[:, 6], rows[:, 7]

        # Helper to slice and concat
        def concat_slices(data_arr, starts, lengths):
            if data_arr is None:
                return None
            slices = [data_arr[s : s + l] for s, l in zip(starts, lengths)]
            if not slices:
                return None
            return np.concatenate(slices, axis=0)

        # Simple features
        b_atoms = concat_slices(self.dataset.data["atom_types"], n_start, n_len)
        b_dists = concat_slices(self.dataset.data["edge_dist"], e_start, e_len)
        b_angles = concat_slices(self.dataset.data["triplet_angle"], t_start, t_len)
        b_ctypes = concat_slices(self.dataset.data["coupling_types"], c_start, c_len)
        b_cvals = concat_slices(self.dataset.data["coupling_values"], c_start, c_len)
        b_shield = concat_slices(self.dataset.data["aux_shielding"], n_start, n_len)
        b_charge = concat_slices(self.dataset.data["aux_charges"], n_start, n_len)

        # Index adjustments (Global -> Batch Local)
        # We need to re-index edges/triplets because they point to global indices in the monolithic array
        # or relative indices within the molecule. The stored arrays are usually relative to the molecule start
        # if processed individually, but here we stored them raw.
        # Wait, process_molecule_task returns indices relative to 0 for that molecule.
        # So when we concat, we just need to shift them by the batch-local offset.

        node_offsets = np.cumsum(np.concatenate(([0], n_len[:-1])))
        edge_offsets = np.cumsum(np.concatenate(([0], e_len[:-1])))

        # Edges
        # self.dataset.data["edge_index"] is stored as (2, Total_Edges).
        # We need to slice (2, len) chunks.
        if self.dataset.data["edge_index"] is not None:
            raw_edges = [
                self.dataset.data["edge_index"][:, s : s + l]
                for s, l in zip(e_start, e_len)
            ]
            b_edges = []
            for edges, local_n in zip(raw_edges, node_offsets):
                # Edges are already 0-indexed per molecule. Just add local batch offset.
                b_edges.append(edges + local_n)
            b_edges = np.concatenate(b_edges, axis=1) if b_edges else np.zeros((2, 0))
        else:
            b_edges = None

        # Triplets
        if self.dataset.data["triplet_index"] is not None:
            raw_triplets = [
                self.dataset.data["triplet_index"][:, s : s + l]
                for s, l in zip(t_start, t_len)
            ]
            b_triplets = []
            for triplets, local_e in zip(raw_triplets, edge_offsets):
                # Triplets index into edges. Add local edge offset.
                b_triplets.append(triplets + local_e)
            b_triplets = (
                np.concatenate(b_triplets, axis=1) if b_triplets else np.zeros((2, 0))
            )
        else:
            b_triplets = None

        # Coupling Nodes
        if self.dataset.data["coupling_node_indices"] is not None:
            raw_c_nodes = [
                self.dataset.data["coupling_node_indices"][s : s + l]
                for s, l in zip(c_start, c_len)
            ]
            b_c_nodes = []
            for cnodes, local_n in zip(raw_c_nodes, node_offsets):
                b_c_nodes.append(cnodes + local_n)
            b_c_nodes = (
                np.concatenate(b_c_nodes, axis=0) if b_c_nodes else np.zeros((0, 2))
            )
        else:
            b_c_nodes = None

        # Coupling Edges
        if self.dataset.data["coupling_edge_indices"] is not None:
            raw_c_edges = [
                self.dataset.data["coupling_edge_indices"][s : s + l]
                for s, l in zip(c_start, c_len)
            ]
            b_c_edges = []
            for cedges, local_e in zip(raw_c_edges, edge_offsets):
                b_c_edges.append(cedges + local_e)
            b_c_edges = (
                np.concatenate(b_c_edges, axis=0) if b_c_edges else np.zeros((0,))
            )
        else:
            b_c_edges = None

        # To Device
        def to_dev(arr, dtype=torch.float32):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray) and arr.size == 0:
                return torch.tensor(arr, device=self.device, dtype=dtype)
            return torch.tensor(arr, device=self.device, dtype=dtype)

        return {
            "atom_types": to_dev(b_atoms, torch.long),
            "edge_index": to_dev(b_edges, torch.long),
            "edge_dist": to_dev(b_dists, torch.float32),
            "triplet_index": to_dev(b_triplets, torch.long),
            "triplet_angle": to_dev(b_angles, torch.float32),
            "coupling_node_indices": to_dev(b_c_nodes, torch.long),
            "coupling_edge_indices": to_dev(b_c_edges, torch.long),
            "coupling_types": to_dev(b_ctypes, torch.long),
            "coupling_values": to_dev(b_cvals, torch.float32),
            "aux_shielding": to_dev(b_shield, torch.float32),
            "aux_charges": to_dev(b_charge, torch.float32),
        }


def get_dataloaders(batch_size, device="cpu", load_cached_data=True):
    """
    Constructs data loaders for train, validation, and test sets.

    Args:
        batch_size (int): Number of molecules per batch.
        device (str): Device to move tensors to ('cpu' or 'cuda').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Process or Load Data
    data_dir = preprocess_data(load_cached_data)

    # 2. Initialize Datasets
    train_ds = SOADataset("train", data_dir)
    val_ds = SOADataset("val", data_dir)
    test_ds = SOADataset("test", data_dir)

    # 3. Initialize Loaders
    train_loader = SOALoader(train_ds, batch_size, shuffle=True, device=device)
    val_loader = SOALoader(val_ds, batch_size, shuffle=False, device=device)
    test_loader = SOALoader(test_ds, batch_size, shuffle=False, device=device)

    return train_loader, val_loader, test_loader
