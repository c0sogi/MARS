import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from ase.io import read
from ase import neighborlist
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class GaussianDistance(object):
    """
    Expands the distance between atoms using a Gaussian basis set.
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Args:
            dmin (float): Minimum distance.
            dmax (float): Maximum distance.
            step (float): Step size for the basis.
            var (float): Variance of the Gaussian. If None, calculated from step.
        """
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance expansion.

        Args:
            distances (np.array): Array of distances.

        Returns:
            np.array: Expanded distances (N_edges, N_bins).
        """
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class StandardScaler:
    """
    Standardizes targets by removing the mean and scaling to unit variance.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data * self.std) + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]


class CrystalGraphDataset(Dataset):
    """
    PyTorch Dataset for crystal graphs.
    """

    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load pre-processed data from cache.
        """
        self.mode = mode
        self.metadata = pd.read_csv(metadata_path)

        # Debugging: Sample subset if configured
        if Config.DEBUG_SAMPLE_SIZE is not None:
            self.metadata = self.metadata.head(Config.DEBUG_SAMPLE_SIZE)

        self.cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_graphs.npz")

        # Load or process data
        self.data = self._load_or_process_data(load_cached_data)

        # Handle scaling
        self.scaler = StandardScaler()
        if mode == "train":
            # Fit scaler on training targets and save
            targets = self.data["targets"]
            self.scaler.fit(targets)
            self.scaler.save(Config.TARGET_SCALER_PATH)
        else:
            # Load scaler for val/test
            if os.path.exists(Config.TARGET_SCALER_PATH):
                self.scaler.load(Config.TARGET_SCALER_PATH)
            else:
                # Fallback if scaler doesn't exist (e.g. running inference only)
                # In a real pipeline, we'd ensure train runs first.
                print("Warning: Target scaler not found. Using identity scaling.")
                self.scaler.mean = np.zeros(len(Config.TARGET_COLS))
                self.scaler.std = np.ones(len(Config.TARGET_COLS))

    def _load_or_process_data(self, load_cached):
        """
        Loads data from cache or processes it from scratch.
        """
        if load_cached and os.path.exists(self.cache_path):
            print(f"Loading cached {self.mode} data from {self.cache_path}...")
            try:
                # Allow pickle=True is often needed for object arrays, but we try to avoid it.
                # Here we use standard numpy arrays.
                loaded = np.load(self.cache_path)
                return dict(loaded)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        print(f"Processing {self.mode} data...")
        return self._process_data()

    def _process_data(self):
        """
        Reads geometry files and constructs graphs.
        Returns a dictionary of concatenated arrays with offsets.
        """
        all_atom_fea = []
        all_edge_index = []
        all_edge_fea = []
        all_targets = []
        all_ids = []

        # Offsets to reconstruct individual graphs
        atom_offsets = [0]
        edge_offsets = [0]

        total_atoms = 0
        total_edges = 0

        for idx, row in self.metadata.iterrows():
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            material_id = row["id"]

            # Read structure
            try:
                # Cite debug_lesson_1: Explicitly Define Parsers When File Extensions Are Misleading
                atoms = read(file_path, format="aims")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Get node features (atomic numbers)
            # We use atomic numbers directly; embedding layer in model handles the rest
            atom_numbers = atoms.get_atomic_numbers()
            n_atoms = len(atom_numbers)

            # Construct graph using neighbor list
            # Use neighbor_list function for vectorized neighbor search
            # 'i': source index, 'j': target index, 'd': distance
            i_indices, j_indices, distances = neighborlist.neighbor_list(
                "ijd", atoms, Config.CUTOFF_RADIUS, self_interaction=False
            )

            # Filter by max neighbors if necessary (optional optimization)
            # Here we keep all within cutoff as per strategy

            # Store data
            all_atom_fea.append(atom_numbers)

            # Edge index: [2, n_edges]
            edge_index = np.vstack((i_indices, j_indices))
            all_edge_index.append(edge_index)

            all_edge_fea.append(distances)

            # Targets
            if self.mode != "test":
                targets = row[Config.TARGET_COLS].values.astype(np.float32)
                all_targets.append(targets)
            else:
                # Dummy targets for test
                all_targets.append(np.zeros(len(Config.TARGET_COLS), dtype=np.float32))

            all_ids.append(material_id)

            total_atoms += n_atoms
            total_edges += len(distances)

            atom_offsets.append(total_atoms)
            edge_offsets.append(total_edges)

        # Concatenate everything into large arrays
        data = {
            "atom_fea": np.concatenate(all_atom_fea).astype(np.int64),
            "edge_index": np.concatenate(all_edge_index, axis=1).astype(np.int64),
            "edge_fea": np.concatenate(all_edge_fea).astype(np.float32),
            "targets": np.vstack(all_targets).astype(np.float32),
            "ids": np.array(all_ids, dtype=np.int64),
            "atom_offsets": np.array(atom_offsets, dtype=np.int64),
            "edge_offsets": np.array(edge_offsets, dtype=np.int64),
        }

        # Save to cache
        np.savez(self.cache_path, **data)
        print(f"Saved processed {self.mode} data to {self.cache_path}")

        return data

    def __len__(self):
        return len(self.data["ids"])

    def __getitem__(self, idx):
        """
        Returns a single graph data object.
        """
        # Extract graph using offsets
        atom_start = self.data["atom_offsets"][idx]
        atom_end = self.data["atom_offsets"][idx + 1]

        edge_start = self.data["edge_offsets"][idx]
        edge_end = self.data["edge_offsets"][idx + 1]

        atom_fea = torch.LongTensor(self.data["atom_fea"][atom_start:atom_end])
        edge_index = torch.LongTensor(self.data["edge_index"][:, edge_start:edge_end])
        edge_fea = torch.Tensor(self.data["edge_fea"][edge_start:edge_end])

        # Get raw target
        target = self.data["targets"][idx]

        # Normalize target if not test
        if self.mode != "test":
            target = self.scaler.transform(target.reshape(1, -1)).flatten()

        target = torch.Tensor(target)
        material_id = self.data["ids"][idx]

        return atom_fea, edge_index, edge_fea, target, material_id


def collate_graphs(batch):
    """
    Collate function to batch graphs into a single disjoint graph.

    Args:
        batch: List of tuples (atom_fea, edge_index, edge_fea, target, material_id)

    Returns:
        batch_atom_fea: (Total_atoms, )
        batch_edge_index: (2, Total_edges)
        batch_edge_fea: (Total_edges, )
        batch_batch_index: (Total_atoms, ) - mapping atoms to batch index
        batch_targets: (Batch_size, n_targets)
        batch_ids: (Batch_size, )
    """
    atom_feas = []
    edge_indices = []
    edge_feas = []
    targets = []
    ids = []
    batch_indices = []

    num_atoms_running = 0

    for i, (atom_fea, edge_index, edge_fea, target, mat_id) in enumerate(batch):
        # Node features
        atom_feas.append(atom_fea)

        # Edge indices (shifted)
        edge_indices.append(edge_index + num_atoms_running)

        # Edge features
        edge_feas.append(edge_fea)

        # Targets and IDs
        targets.append(target)
        ids.append(mat_id)

        # Batch index (which graph does this atom belong to?)
        n_atoms = atom_fea.shape[0]
        batch_indices.append(torch.full((n_atoms,), i, dtype=torch.long))

        num_atoms_running += n_atoms

    # Concatenate
    batch_atom_fea = torch.cat(atom_feas, dim=0)
    batch_edge_index = torch.cat(edge_indices, dim=1)
    batch_edge_fea = torch.cat(edge_feas, dim=0)
    batch_batch_index = torch.cat(batch_indices, dim=0)
    batch_targets = torch.stack(targets, dim=0)
    batch_ids = torch.tensor(ids, dtype=torch.long)

    return (
        batch_atom_fea,
        batch_edge_index,
        batch_edge_fea,
        batch_batch_index,
        batch_targets,
        batch_ids,
    )
