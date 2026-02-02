import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import ase.io
from ase.neighborlist import neighbor_list

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_20"
METADATA_DIR = "./metadata"


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=60):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def process_structure(file_path, targets=None, cutoff=5.0, smearing_model=None):
    """
    Parses an .xyz file, computes PBC-aware neighbor list, and constructs a PyG Data object.
    """
    # Load structure using ASE
    full_path = os.path.join(INPUT_DIR, file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Structure file not found: {full_path}")

    # Cite debug_lesson_1: Explicitly Define Parsers When File Extensions Are Misleading
    atoms = ase.io.read(full_path, format="aims")

    # Node features: Atomic numbers
    # We map atomic numbers to a 0-based index or use them directly if the embedding layer handles it.
    # Common elements in this dataset: Al(13), Ga(31), In(49), O(8).
    # We will use atomic numbers directly and expect the model to handle the embedding lookup
    # (e.g., max atomic number is < 100).
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Compute neighbor list with PBC
    # 'i' : first atom index
    # 'j' : second atom index
    # 'd' : distance
    # 'D' : distance vector
    # 'S' : shift vector (not strictly needed here but implied by PBC)
    i, j, d = neighbor_list("ijd", atoms, cutoff)

    # Edge indices
    edge_index = torch.stack([torch.from_numpy(i), torch.from_numpy(j)], dim=0).to(
        torch.long
    )

    # Edge features: RBF expansion of distances
    distances = torch.from_numpy(d).to(torch.float)

    if smearing_model is None:
        smearing_model = GaussianSmearing(0.0, cutoff, 60)

    edge_attr = smearing_model(distances)

    # Targets
    y = None
    if targets is not None:
        y = torch.tensor(targets, dtype=torch.float).view(1, -1)

    data = Data(x=z, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


def save_graphs_to_npz(data_list, path):
    """
    Saves a list of PyG Data objects to a compressed npz file using a CSR-like format
    to avoid pickling and ensure efficiency.
    """
    # Concatenate all attributes
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_y = []

    node_ptr = [0]
    edge_ptr = [0]

    for data in data_list:
        all_x.append(data.x.numpy())
        all_edge_index.append(data.edge_index.numpy())
        all_edge_attr.append(data.edge_attr.numpy())

        if data.y is not None:
            all_y.append(data.y.numpy())
        else:
            # Placeholder for test set if needed, though usually we handle None
            all_y.append(np.array([[np.nan, np.nan]]))

        node_ptr.append(node_ptr[-1] + data.x.shape[0])
        edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

    np.savez_compressed(
        path,
        x=np.concatenate(all_x),
        edge_index=np.concatenate(all_edge_index, axis=1),
        edge_attr=np.concatenate(all_edge_attr),
        y=np.concatenate(all_y),
        node_ptr=np.array(node_ptr),
        edge_ptr=np.array(edge_ptr),
    )


def load_graphs_from_npz(path):
    """
    Loads graphs from the CSR-like npz format.
    """
    data = np.load(path)
    x = torch.from_numpy(data["x"])
    edge_index = torch.from_numpy(data["edge_index"])
    edge_attr = torch.from_numpy(data["edge_attr"])
    y = torch.from_numpy(data["y"])
    node_ptr = data["node_ptr"]
    edge_ptr = data["edge_ptr"]

    data_list = []
    num_graphs = len(node_ptr) - 1

    for i in range(num_graphs):
        n_start, n_end = node_ptr[i], node_ptr[i + 1]
        e_start, e_end = edge_ptr[i], edge_ptr[i + 1]

        # Slice
        g_x = x[n_start:n_end]
        g_edge_index = edge_index[:, e_start:e_end]
        g_edge_attr = edge_attr[e_start:e_end]
        g_y = y[i : i + 1]

        # Adjust edge indices to be 0-based for the subgraph
        # The stored edge indices are likely 0-based relative to the specific graph
        # because neighbor_list returns indices relative to the atoms object.
        # However, let's verify. process_structure returns 0-based indices for that graph.
        # So we don't need to offset them by n_start unless we were doing batching manually.
        # Here we are reconstructing individual Data objects.

        # Handle NaN y for test set
        if torch.isnan(g_y).any():
            g_y = None

        data_obj = Data(x=g_x, edge_index=g_edge_index, edge_attr=g_edge_attr, y=g_y)
        data_list.append(data_obj)

    return data_list


class CrystalDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True, sample_size=None):
        super().__init__()
        self.mode = mode
        self.cache_path = os.path.join(CACHE_DIR, f"{mode}_graphs.npz")
        self.data_list = []

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        loaded = False
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached {mode} data from {self.cache_path}...")
                self.data_list = load_graphs_from_npz(self.cache_path)
                loaded = True
                print(f"Successfully loaded {len(self.data_list)} graphs.")
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
                loaded = False

        if not loaded:
            # Load metadata
            meta_file = os.path.join(METADATA_DIR, f"{mode}_metadata.csv")
            if not os.path.exists(meta_file):
                raise FileNotFoundError(f"Metadata file not found: {meta_file}")

            df = pd.read_csv(meta_file)

            # Subsample if requested (for debugging)
            if sample_size is not None:
                df = df.iloc[:sample_size]

            print(f"Processing {len(df)} structures for {mode} set...")

            smearing = GaussianSmearing(0.0, 5.0, 60)

            for _, row in df.iterrows():
                # Extract targets if available
                targets = None
                if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
                    targets = [
                        row["formation_energy_ev_natom"],
                        row["bandgap_energy_ev"],
                    ]

                try:
                    data = process_structure(
                        row["file_path"],
                        targets=targets,
                        cutoff=5.0,
                        smearing_model=smearing,
                    )
                    # Attach ID for submission tracking
                    data.id = torch.tensor([row["id"]], dtype=torch.long)
                    self.data_list.append(data)
                except Exception as e:
                    print(f"Error processing {row['file_path']}: {e}")

            # Save to cache
            print(f"Saving processed data to {self.cache_path}...")
            save_graphs_to_npz(self.data_list, self.cache_path)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_dataloaders(
    batch_size=32, num_workers=0, load_cached_data=True, sample_size=None
):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    train_dataset = CrystalDataset(
        mode="train", load_cached_data=load_cached_data, sample_size=sample_size
    )
    val_dataset = CrystalDataset(
        mode="val", load_cached_data=load_cached_data, sample_size=sample_size
    )
    test_dataset = CrystalDataset(
        mode="test", load_cached_data=load_cached_data, sample_size=sample_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    # Validation and Test loaders usually don't need shuffling, but for validation it doesn't hurt.
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
