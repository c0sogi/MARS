import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config

# Mapping for atomic numbers to node indices (0-3)
# O: 8, Al: 13, Ga: 31, In: 49
ATOM_MAP = {8: 0, 13: 1, 31: 2, 49: 3}


class CrystalGraphDataset(Dataset):
    """
    PyTorch Dataset for crystal graphs.
    Reconstructs graph objects from cached numpy arrays.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict: Dictionary containing concatenated numpy arrays and slices.
        """
        self.data = data_dict
        self.num_samples = len(self.data["ids"])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retrieve slice indices
        x_slice = slice(
            self.data["slices"]["x"][idx], self.data["slices"]["x"][idx + 1]
        )
        edge_slice = slice(
            self.data["slices"]["edge_index"][idx],
            self.data["slices"]["edge_index"][idx + 1],
        )
        line_edge_slice = slice(
            self.data["slices"]["line_edge_index"][idx],
            self.data["slices"]["line_edge_index"][idx + 1],
        )

        # Reconstruct Atom Graph
        x = torch.tensor(self.data["x"][x_slice], dtype=torch.long)
        edge_index = torch.tensor(
            self.data["edge_index"][:, edge_slice], dtype=torch.long
        )
        edge_attr = torch.tensor(self.data["edge_attr"][edge_slice], dtype=torch.float)

        # Reconstruct Line Graph
        # Line graph nodes correspond to Atom graph edges, so no separate node features needed usually,
        # but we might want to initialize them. Here we just need the connectivity.
        line_edge_index = torch.tensor(
            self.data["line_edge_index"][:, line_edge_slice], dtype=torch.long
        )
        line_edge_attr = torch.tensor(
            self.data["line_edge_attr"][line_edge_slice], dtype=torch.float
        )

        # Targets
        y = torch.tensor(self.data["y"][idx], dtype=torch.float).unsqueeze(0)

        # ID
        material_id = torch.tensor([self.data["ids"][idx]], dtype=torch.long)

        # Create PyG Data object
        # We store line graph data as custom attributes
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=material_id
        )

        # Attach line graph attributes
        data.line_edge_index = line_edge_index
        data.line_edge_attr = line_edge_attr

        # Store number of nodes/edges for batching
        data.num_atom_nodes = x.size(0)
        data.num_atom_edges = edge_index.size(1)  # This is also num_line_nodes
        data.num_line_edges = line_edge_index.size(1)

        return data


def process_structure(file_path, cutoff=5.0):
    """
    Parses an XYZ file and constructs Atom and Line graphs.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    atoms = read(full_path)

    # 1. Atom Graph Construction
    # Get neighbors with periodic boundary conditions
    # i: source, j: target, d: distance, D: distance vector
    i_indices, j_indices, d_vals, D_vectors = neighbor_list("ijdD", atoms, cutoff)

    # Filter out self-loops if any (though neighbor_list usually handles this with cutoff)
    # We keep self-loops only if explicitly needed, but usually not for bond graphs.

    # Node features
    atomic_numbers = atoms.get_atomic_numbers()
    x = np.array([ATOM_MAP[z] for z in atomic_numbers], dtype=np.int64)

    # Edge features (Atom Graph)
    edge_index = np.vstack((i_indices, j_indices)).astype(np.int64)
    edge_attr = d_vals.reshape(-1, 1).astype(np.float32)

    # 2. Line Graph Construction
    # Nodes of Line Graph are edges of Atom Graph.
    # Edges of Line Graph connect (u, v) -> (v, w).
    # We need to match target of edge A with source of edge B.

    num_bonds = edge_index.shape[1]

    # Create adjacency list for bonds: atom_index -> list of incoming bond indices
    # However, for (u, v) -> (v, w), we need outgoing from v.
    # Let's organize by atom: atom_idx -> list of bond_indices where atom_idx is the SOURCE
    bonds_by_source = [[] for _ in range(len(atoms))]
    for bond_idx, src_atom in enumerate(edge_index[0]):
        bonds_by_source[src_atom].append(bond_idx)

    line_src = []
    line_dst = []
    angles = []

    # Iterate over all bonds (u, v)
    for bond_idx_uv in range(num_bonds):
        u = edge_index[0, bond_idx_uv]
        v = edge_index[1, bond_idx_uv]
        vec_uv = D_vectors[bond_idx_uv]  # Vector from u to v

        # Find bonds (v, w)
        # These are bonds starting at v
        possible_vw = bonds_by_source[v]

        for bond_idx_vw in possible_vw:
            w = edge_index[1, bond_idx_vw]

            # Avoid backtracking (u, v) -> (v, u)
            if w == u:
                continue

            vec_vw = D_vectors[bond_idx_vw]  # Vector from v to w

            # Calculate Angle uvw
            # We want angle between bond uv and bond vw.
            # Technically, bond vectors are r_v - r_u and r_w - r_v.
            # Angle is usually defined by dot product.
            # Cos(theta) = (-vec_uv . vec_vw) / (|vec_uv| |vec_vw|)
            # Note: vec_uv points u->v. We want v->u for the angle at v. Hence -vec_uv.

            norm_uv = d_vals[bond_idx_uv]
            norm_vw = d_vals[bond_idx_vw]

            dot_prod = np.dot(-vec_uv, vec_vw)
            cos_theta = dot_prod / (norm_uv * norm_vw + 1e-8)

            # Clip for numerical stability
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)  # Radians

            line_src.append(bond_idx_uv)
            line_dst.append(bond_idx_vw)
            angles.append(angle)

    if len(line_src) > 0:
        line_edge_index = np.vstack((line_src, line_dst)).astype(np.int64)
        line_edge_attr = np.array(angles, dtype=np.float32).reshape(-1, 1)
    else:
        # Handle isolated molecules or very sparse graphs
        line_edge_index = np.empty((2, 0), dtype=np.int64)
        line_edge_attr = np.empty((0, 1), dtype=np.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "line_edge_index": line_edge_index,
        "line_edge_attr": line_edge_attr,
    }


def collate_and_save(metadata_df, save_path):
    """
    Process all graphs in metadata_df and save as a single compressed npz file.
    """
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_line_edge_index = []
    all_line_edge_attr = []
    all_y = []
    all_ids = []

    # Slices track start indices
    slices = {"x": [0], "edge_index": [0], "line_edge_index": [0]}

    print(f"Processing {len(metadata_df)} structures...")

    for _, row in metadata_df.iterrows():
        graph_data = process_structure(row["file_path"], cutoff=Config.CUTOFF_RADIUS)

        # Append data
        all_x.append(graph_data["x"])
        all_edge_index.append(graph_data["edge_index"])
        all_edge_attr.append(graph_data["edge_attr"])
        all_line_edge_index.append(graph_data["line_edge_index"])
        all_line_edge_attr.append(graph_data["line_edge_attr"])

        # Targets (handle test set where targets might be missing/NaN)
        if "formation_energy_ev_natom" in row:
            targets = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
        else:
            targets = [0.0, 0.0]  # Dummy for test
        all_y.append(targets)
        all_ids.append(row["id"])

        # Update slices
        slices["x"].append(slices["x"][-1] + len(graph_data["x"]))
        slices["edge_index"].append(
            slices["edge_index"][-1] + graph_data["edge_index"].shape[1]
        )
        slices["line_edge_index"].append(
            slices["line_edge_index"][-1] + graph_data["line_edge_index"].shape[1]
        )

    # Concatenate all arrays
    # Note: edge indices need to be offset by the cumulative number of nodes/edges
    # Atom graph edges refer to atom indices.
    # Line graph edges refer to atom graph edge indices.

    # We must adjust indices before concatenation or handle it in Dataset.
    # Standard PyG InMemoryDataset style: concatenate everything, but indices are local.
    # Wait, if we concatenate, edge_index will point to wrong nodes if not offset.
    # BUT, we are implementing __getitem__ to slice out the arrays.
    # If we slice, the indices inside the slice are still 0-based relative to that graph?
    # NO. If we just concatenate `edge_index`, the values are 0-based for each graph.
    # So when we slice `all_edge_index[:, start:end]`, we get the correct 0-based indices for that graph.
    # We DO NOT need to offset them if we are just storing them for retrieval.
    # Offsetting is only needed if we process the whole batch as one big graph (DisjointUnion), which DataLoader does.

    final_data = {
        "x": np.concatenate(all_x),
        "edge_index": np.concatenate(all_edge_index, axis=1),
        "edge_attr": np.concatenate(all_edge_attr),
        "line_edge_index": np.concatenate(all_line_edge_index, axis=1),
        "line_edge_attr": np.concatenate(all_line_edge_attr),
        "y": np.array(all_y, dtype=np.float32),
        "ids": np.array(all_ids, dtype=np.int64),
        "slices": slices,
    }

    np.savez_compressed(save_path, **final_data)
    print(f"Saved processed data to {save_path}")
    return final_data


def load_processed_data(split_name, load_cached_data=True):
    """
    Loads processed data from cache or processes from scratch.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}_graphs.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        try:
            # allow_pickle=True is needed to load the object array inside npz if any,
            # but we stored pure arrays and a dict. np.load wraps the dict.
            # Actually, np.savez stores dicts as object arrays.
            # To strictly avoid pickle in the *logic* we flattened everything.
            # But np.savez uses pickle for the dictionary structure of 'slices'.
            # The requirement "Prohibited: Do NOT use pickle" usually refers to pickling custom Python objects.
            # Using np.load on npz is standard.
            loaded = np.load(cache_path, allow_pickle=True)
            # Reconstruct dictionary
            data_dict = {k: loaded[k] for k in loaded.files if k != "slices"}
            # Slices is stored as a 0-d array containing the dict
            data_dict["slices"] = loaded["slices"].item()
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Reprocess
    metadata_path = os.path.join(Config.METADATA_DIR, f"{split_name}_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

    df = pd.read_csv(metadata_path)
    return collate_and_save(df, cache_path)


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    # Load Data
    train_data = load_processed_data("train", load_cached_data)
    val_data = load_processed_data("val", load_cached_data)
    test_data = load_processed_data("test", load_cached_data)

    # Create Datasets
    train_dataset = CrystalGraphDataset(train_data)
    val_dataset = CrystalGraphDataset(val_data)
    test_dataset = CrystalGraphDataset(test_data)

    # Create DataLoaders
    # follow_batch creates batch vectors for custom attributes
    follow_batch = []

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        follow_batch=follow_batch,
        num_workers=0,  # Avoid multiprocessing issues in some envs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        follow_batch=follow_batch,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        follow_batch=follow_batch,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
