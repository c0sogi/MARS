import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import KDTree

# Import provided library components
from library.config import Config
from library.utils import seed_everything, TargetScaler, MetricLogger
from library.layers import get_model

# ==========================================
# Data Processing & Loading (Flattened SoA)
# ==========================================


class MoleculeBatcher:
    """
    Efficiently batches flattened Structure-of-Arrays data from disk to GPU.
    Avoids the overhead of list-of-objects by slicing monolithic numpy arrays.
    """

    def __init__(self, data_dir, split, batch_size, shuffle=False, device="cpu"):
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        # Load molecule map: [n_start, n_len, e_start, e_len, t_start, t_len, c_start, c_len]
        self.mol_map = np.load(os.path.join(data_dir, f"{split}_mol_map.npy"))
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
        rows = self.mol_map[batch_indices]
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
        b_atoms = concat_slices(self.data["atom_types"], n_start, n_len)
        b_dists = concat_slices(self.data["edge_dist"], e_start, e_len)
        b_angles = concat_slices(self.data["triplet_angle"], t_start, t_len)
        b_ctypes = concat_slices(self.data["coupling_types"], c_start, c_len)
        b_cvals = concat_slices(self.data["coupling_values"], c_start, c_len)
        b_shield = concat_slices(self.data["aux_shielding"], n_start, n_len)
        b_charge = concat_slices(self.data["aux_charges"], n_start, n_len)

        # Index adjustments (Global -> Batch Local)
        node_offsets = np.cumsum(np.concatenate(([0], n_len[:-1])))
        edge_offsets = np.cumsum(np.concatenate(([0], e_len[:-1])))

        # Edges
        raw_edges = [
            self.data["edge_index"][:, s : s + l] for s, l in zip(e_start, e_len)
        ]
        b_edges = []
        for edges, global_n, local_n in zip(raw_edges, n_start, node_offsets):
            b_edges.append(edges - global_n + local_n)
        b_edges = np.concatenate(b_edges, axis=1) if b_edges else np.zeros((2, 0))

        # Triplets
        raw_triplets = [
            self.data["triplet_index"][:, s : s + l] for s, l in zip(t_start, t_len)
        ]
        b_triplets = []
        for triplets, global_e, local_e in zip(raw_triplets, e_start, edge_offsets):
            b_triplets.append(triplets - global_e + local_e)
        b_triplets = (
            np.concatenate(b_triplets, axis=1) if b_triplets else np.zeros((2, 0))
        )

        # Coupling Nodes
        raw_c_nodes = [
            self.data["coupling_node_indices"][s : s + l]
            for s, l in zip(c_start, c_len)
        ]
        b_c_nodes = []
        for cnodes, global_n, local_n in zip(raw_c_nodes, n_start, node_offsets):
            b_c_nodes.append(cnodes - global_n + local_n)
        b_c_nodes = np.concatenate(b_c_nodes, axis=0) if b_c_nodes else np.zeros((0, 2))

        # Coupling Edges
        raw_c_edges = [
            self.data["coupling_edge_indices"][s : s + l]
            for s, l in zip(c_start, c_len)
        ]
        b_c_edges = []
        for cedges, global_e, local_e in zip(raw_c_edges, e_start, edge_offsets):
            b_c_edges.append(cedges - global_e + local_e)
        b_c_edges = np.concatenate(b_c_edges, axis=0) if b_c_edges else np.zeros((0,))

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

    def __len__(self):
        return (self.num_mols + self.batch_size - 1) // self.batch_size


def process_molecule_task(args):
    """Worker function to process a single molecule into graph arrays."""
    mol_name, struct_df, meta_df, aux_data = args

    # Nodes
    atoms = struct_df["atom"].map(Config.ATOM_MAP).values.astype(np.int64)
    coords = struct_df[["x", "y", "z"]].values.astype(np.float32)
    num_nodes = len(atoms)

    # Edges (Radius Graph)
    tree = KDTree(coords)
    pairs = tree.query_pairs(Config.CUTOFF)

    u_list, v_list, dist_list = [], [], []
    for i, j in pairs:
        d = np.linalg.norm(coords[i] - coords[j])
        u_list.extend([i, j])
        v_list.extend([j, i])
        dist_list.extend([d, d])

    edge_index = np.array([u_list, v_list], dtype=np.int64)
    edge_dist = np.array(dist_list, dtype=np.float32)
    num_edges = len(dist_list)

    edge_lookup = {(u, v): idx for idx, (u, v) in enumerate(zip(u_list, v_list))}

    # Triplets (k -> j -> i)
    # Build adjacency
    incoming = {n: [] for n in range(num_nodes)}
    outgoing = {n: [] for n in range(num_nodes)}
    for idx, (u, v) in enumerate(zip(u_list, v_list)):
        incoming[v].append(idx)
        outgoing[u].append(idx)

    trip_src, trip_dst, trip_angle = [], [], []
    for j in range(num_nodes):
        for e_in in incoming[j]:
            k = u_list[e_in]
            vec_kj = coords[j] - coords[k]

            for e_out in outgoing[j]:
                i = v_list[e_out]
                if k == i:
                    continue

                vec_ji = coords[i] - coords[j]

                # Cosine angle
                denom = edge_dist[e_in] * edge_dist[e_out] + 1e-9
                cos = np.dot(vec_kj, vec_ji) / denom

                trip_src.append(e_in)
                trip_dst.append(e_out)
                trip_angle.append(np.clip(cos, -1.0, 1.0))

    triplet_index = np.array([trip_src, trip_dst], dtype=np.int64)
    triplet_angle = np.array(trip_angle, dtype=np.float32)

    # Couplings
    c_types, c_values, c_node_indices, c_edge_indices = [], [], [], []
    if meta_df is not None and not meta_df.empty:
        c_node_indices = meta_df[["atom_index_0", "atom_index_1"]].values.astype(
            np.int64
        )
        c_types = [Config.COUPLING_MAP[t] for t in meta_df["type"]]
        c_values = meta_df.get(
            "scalar_coupling_constant", pd.Series([0] * len(meta_df))
        ).values.astype(np.float32)

        for u, v in c_node_indices:
            c_edge_indices.append(edge_lookup.get((u, v), 0))

    c_types = np.array(c_types, dtype=np.int64)
    c_edge_indices = np.array(c_edge_indices, dtype=np.int64)

    # Aux
    aux_s = (
        aux_data.get("shielding")[
            ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
        ].values.astype(np.float32)
        if "shielding" in aux_data
        else None
    )
    aux_c = (
        aux_data.get("charges")["mulliken_charge"]
        .values.astype(np.float32)
        .reshape(-1, 1)
        if "charges" in aux_data
        else None
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


def prepare_data(load_cached_data=True):
    cache_dir = os.path.join(Config.WORKING_DIR, "processed")
    os.makedirs(cache_dir, exist_ok=True)

    if load_cached_data and os.path.exists(
        os.path.join(cache_dir, "train_mol_map.npy")
    ):
        print("Loading cached processed data...")
        return cache_dir

    print("Processing data from scratch...")
    struct_df = pd.read_csv(Config.STRUCTURES_CSV)
    shielding = pd.read_csv(Config.MAGNETIC_SHIELDING_PATH)
    charges = pd.read_csv(Config.MULLIKEN_CHARGES_PATH)

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
        mol_names = sorted(meta_grp.groups.keys())  # Ensure deterministic order

        tasks = []
        for m in mol_names:
            aux = {}
            if split_name != "test":
                if m in shield_grp.groups:
                    aux["shielding"] = shield_grp.get_group(m)
                if m in charge_grp.groups:
                    aux["charges"] = charge_grp.get_group(m)
            tasks.append((m, struct_grp.get_group(m), meta_grp.get_group(m), aux))

        print(f"Processing {split_name} ({len(tasks)} molecules)...")
        results = []
        with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as pool:
            for res in pool.map(process_molecule_task, tasks):
                results.append(res)

        # Aggregate and Save
        arrays = {k: [] for k in results[0].keys() if not k.startswith("num_")}
        mol_map = []
        ptrs = {"n": 0, "e": 0, "t": 0, "c": 0}

        for r in results:
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

        np.save(
            os.path.join(cache_dir, f"{split_name}_mol_map.npy"),
            np.array(mol_map, dtype=np.int64),
        )
        for k, v in arrays.items():
            if v:
                concat_func = (
                    np.concatenate
                    if k != "coupling_node_indices"
                    else lambda x: np.concatenate(x, axis=0)
                )
                # Handle special case for 2D arrays if needed, but np.concatenate works for axis=0 default
                # Explicit axis handling:
                axis = 1 if k in ["edge_index", "triplet_index"] else 0
                if k == "coupling_node_indices":
                    axis = 0
                if len(v) > 0:
                    np.save(
                        os.path.join(cache_dir, f"{split_name}_{k}.npy"),
                        np.concatenate(v, axis=axis),
                    )

    return cache_dir


# ==========================================
# Workflow Execution
# ==========================================


def run_workflow():
    seed_everything(Config.SEED)

    # 1. Data Preparation
    data_dir = prepare_data(load_cached_data=True)

    # 2. Target Scaling
    print("Fitting Scaler...")
    train_vals = np.load(os.path.join(data_dir, "train_coupling_values.npy"))
    train_types = np.load(os.path.join(data_dir, "train_coupling_types.npy"))

    scaler = TargetScaler()
    # Manual fit since we have indices
    for i, t_str in enumerate(Config.COUPLING_TYPES):
        mask = train_types == i
        if np.any(mask):
            vals = train_vals[mask]
            scaler.mean_arr[i] = np.mean(vals)
            scaler.std_arr[i] = np.std(vals)
    scaler.fitted = True

    # 3. Model Setup
    model = get_model(Config()).to(Config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    train_loader = MoleculeBatcher(
        data_dir, "train", Config.BATCH_SIZE, shuffle=True, device=Config.DEVICE
    )
    val_loader = MoleculeBatcher(
        data_dir, "val", Config.BATCH_SIZE, shuffle=False, device=Config.DEVICE
    )

    # 4. Training Loop
    print("Starting Training...")
    best_score = float("inf")

    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch["atom_types"],
                batch["edge_index"],
                batch["edge_dist"],
                batch["triplet_index"],
                batch["triplet_angle"],
                batch["coupling_node_indices"],
                batch["coupling_edge_indices"],
                batch["coupling_types"],
            )

            # Standardize Targets
            means = torch.tensor(scaler.mean_arr, device=Config.DEVICE)[
                batch["coupling_types"]
            ]
            stds = torch.tensor(scaler.std_arr, device=Config.DEVICE)[
                batch["coupling_types"]
            ]
            targets = (batch["coupling_values"] - means) / stds

            loss = nn.functional.l1_loss(out["coupling"].squeeze(), targets)

            if Config.USE_AUXILIARY_HEADS:
                loss += Config.AUX_LOSS_WEIGHT * (
                    nn.functional.l1_loss(out["shielding"], batch["aux_shielding"])
                    + nn.functional.l1_loss(out["charge"], batch["aux_charges"])
                )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        logger = MetricLogger()
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    batch["atom_types"],
                    batch["edge_index"],
                    batch["edge_dist"],
                    batch["triplet_index"],
                    batch["triplet_angle"],
                    batch["coupling_node_indices"],
                    batch["coupling_edge_indices"],
                    batch["coupling_types"],
                )

                means = torch.tensor(scaler.mean_arr, device=Config.DEVICE)[
                    batch["coupling_types"]
                ]
                stds = torch.tensor(scaler.std_arr, device=Config.DEVICE)[
                    batch["coupling_types"]
                ]
                targets = (batch["coupling_values"] - means) / stds

                logger.update(
                    out["coupling"].squeeze(), targets, batch["coupling_types"]
                )

        val_score = logger.compute_metric(scaler)
        print(
            f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val LMAE {val_score:.9f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 5. Submission
    print("Generating Submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    test_loader = MoleculeBatcher(
        data_dir, "test", Config.BATCH_SIZE, shuffle=False, device=Config.DEVICE
    )
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                batch["atom_types"],
                batch["edge_index"],
                batch["edge_dist"],
                batch["triplet_index"],
                batch["triplet_angle"],
                batch["coupling_node_indices"],
                batch["coupling_edge_indices"],
                batch["coupling_types"],
            )
            preds.append(
                scaler.inverse_transform(
                    out["coupling"].squeeze(), batch["coupling_types"]
                )
            )

    all_preds = np.concatenate(preds)

    # Re-align predictions with IDs
    # MoleculeBatcher iterates sorted molecules. We must extract IDs in that same order.
    test_meta = pd.read_csv(Config.TEST_META_PATH)
    test_grp = test_meta.groupby("molecule_name")
    sorted_mols = sorted(test_grp.groups.keys())

    ids_ordered = []
    for m in sorted_mols:
        ids_ordered.append(test_grp.get_group(m)["id"].values)
    ids_ordered = np.concatenate(ids_ordered)

    pred_map = dict(zip(ids_ordered, all_preds))

    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    sub_df["scalar_coupling_constant"] = sub_df["id"].map(pred_map)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved.")


run_workflow()
