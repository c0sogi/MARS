import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from library.config import Config
from library.utils import set_seed


class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        metadata_file,
        structures_file,
        mode="train",
        load_cached=True,
        transform=None,
        pre_transform=None,
    ):
        self.metadata_file = metadata_file
        self.structures_file = structures_file
        self.mode = mode
        self.load_cached = load_cached

        # Define cache path based on mode
        self.cache_path = os.path.join(Config.IDEA_WORK_DIR, f"cached_{mode}_v2.npz")

        # Ensure working directory exists
        os.makedirs(Config.IDEA_WORK_DIR, exist_ok=True)

        super().__init__(root, transform, pre_transform)

        # Load or Process Data
        if self.load_cached and os.path.exists(self.cache_path):
            print(f"Loading cached dataset from {self.cache_path}...")
            self.load_cache()
        else:
            print(f"Processing dataset for {mode}...")
            self.process_raw()
            self.save_cache()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process_raw(self):
        set_seed()

        # 1. Load Structures
        # structures.csv: molecule_name, atom_index, atom, x, y, z
        df_struct = pd.read_csv(self.structures_file)

        # 2. Load Metadata
        # metadata: molecule_name, atom_index_0, atom_index_1, type, [scalar_coupling_constant], id
        df_meta = pd.read_csv(self.metadata_file)

        # 3. Debugging: Sample subset if configured
        if Config.DEBUG_SAMPLE_SIZE is not None:
            # Cite debug_lesson_4: Verify intersection of keys between data sources.
            meta_mols = df_meta["molecule_name"].unique()
            struct_mols = df_struct["molecule_name"].unique()

            # Find valid molecules present in both
            available_mols = np.intersect1d(meta_mols, struct_mols)

            if len(available_mols) == 0:
                raise ValueError(
                    f"No overlap found between metadata and structures for {self.mode}. Check data consistency."
                )

            # Sample from the intersection
            if len(available_mols) > Config.DEBUG_SAMPLE_SIZE:
                selected_mols = available_mols[: Config.DEBUG_SAMPLE_SIZE]
            else:
                selected_mols = available_mols

            # Filter both dataframes
            df_meta = df_meta[df_meta["molecule_name"].isin(selected_mols)].copy()
            df_struct = df_struct[df_struct["molecule_name"].isin(selected_mols)].copy()
            print(
                f"Debug Mode: Sampled {len(selected_mols)} molecules (from intersection)."
            )

        # 4. Pre-processing Structures
        # Sort to ensure atom_index corresponds to array index (0 to N-1)
        df_struct = df_struct.sort_values(["molecule_name", "atom_index"])

        # Group by molecule for efficient iteration
        struct_groups = df_struct.groupby("molecule_name")
        meta_groups = df_meta.groupby("molecule_name")

        data_list = []

        # Iterate over molecules present in metadata
        # Using list(meta_groups) to avoid re-iteration issues
        for mol_name, group_meta in meta_groups:
            if mol_name not in struct_groups.groups:
                continue

            # Get structure data
            group_struct = struct_groups.get_group(mol_name)

            # Node Features: Atom Type
            # Map atom symbol to integer
            atoms = group_struct["atom"].map(Config.ATOM_MAP).values
            x = torch.tensor(atoms, dtype=torch.long)

            # Node Positions
            coords = group_struct[["x", "y", "z"]].values
            pos = torch.tensor(coords, dtype=torch.float)

            # Edge Construction (Structural Graph)
            # Compute pairwise Euclidean distances
            # shape: (N, 3) -> (N, 1, 3) - (1, N, 3) -> (N, N, 3)
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

            # Create edges where distance < cutoff and distance > 0 (exclude self-loops)
            mask = (dist_matrix < Config.CUTOFF_RADIUS) & (dist_matrix > 1e-6)
            row, col = np.where(mask)

            edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)

            # Edge Attributes: Inverse Distance
            distances = dist_matrix[row, col]
            edge_attr = torch.tensor(1.0 / distances, dtype=torch.float).unsqueeze(1)

            # Target / Coupling Data
            # These are the pairs we need to predict for this molecule
            idx0 = group_meta["atom_index_0"].values
            idx1 = group_meta["atom_index_1"].values
            couple_idx = torch.tensor(np.stack([idx0, idx1], axis=1), dtype=torch.long)
            num_couples = torch.tensor([len(idx0)], dtype=torch.long)

            types = group_meta["type"].map(Config.TYPE_MAP).values
            couple_type = torch.tensor(types, dtype=torch.long)

            ids = group_meta["id"].values
            id_tensor = torch.tensor(ids, dtype=torch.long)

            # Target Values
            if "scalar_coupling_constant" in group_meta.columns:
                raw_y = group_meta["scalar_coupling_constant"].values
                # Normalize target
                norm_y = (raw_y - Config.TARGET_MEAN) / Config.TARGET_STD
                y = torch.tensor(norm_y, dtype=torch.float)
            else:
                # For test set, use dummy zeros
                y = torch.zeros(len(group_meta), dtype=torch.float)

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                couple_idx=couple_idx,
                num_couples=num_couples,
                couple_type=couple_type,
                y=y,
                id=id_tensor,
                num_nodes=len(atoms),  # Explicitly set num_nodes
            )
            data_list.append(data)

        print(f"Processed {len(data_list)} graphs.")

        # Collate list of Data objects into internal storage
        if len(data_list) > 0:
            self.data, self.slices = self.collate(data_list)
        else:
            # Handle empty dataset case
            self.data = Data()
            self.slices = {}

    def save_cache(self):
        """
        Saves self.data and self.slices to a .npz file using numpy.
        This avoids pickle on arbitrary objects.
        """
        save_dict = {}

        # 1. Save Data attributes
        # self.data is a Data object. In PyG, iterating it directly yields keys.
        # We must explicitly convert to dict or iterate items to get values.
        if hasattr(self.data, "to_dict"):
            data_map = self.data.to_dict()
        else:
            # Fallback if it's already a dict or behaves like one
            data_map = self.data

        for key, value in data_map.items():
            if torch.is_tensor(value):
                save_dict[f"data_{key}"] = value.numpy()
            else:
                # Fallback for non-tensor attributes (unlikely in this setup)
                pass

        # 2. Save Slices
        # self.slices is a dict: key -> tensor (start/end indices)
        for key, value in self.slices.items():
            if torch.is_tensor(value):
                save_dict[f"slices_{key}"] = value.numpy()

        # Save keys to reconstruct the dictionary structure
        slice_keys = list(self.slices.keys())
        save_dict["slice_keys"] = np.array(slice_keys)

        np.savez(self.cache_path, **save_dict)
        print(f"Dataset cached to {self.cache_path}")

    def load_cache(self):
        """
        Loads dataset from .npz file.
        """
        try:
            loaded = np.load(self.cache_path)

            # Reconstruct Data object
            data_dict = {}
            # We iterate over keys in the loaded file
            for file_key in loaded.files:
                if file_key.startswith("data_"):
                    attr_name = file_key[5:]  # remove 'data_'
                    tensor = torch.from_numpy(loaded[file_key])
                    data_dict[attr_name] = tensor

            self.data = Data.from_dict(data_dict)

            # Reconstruct Slices dictionary
            self.slices = {}
            if "slice_keys" in loaded:
                slice_keys = loaded["slice_keys"]
                for key in slice_keys:
                    slice_file_key = f"slices_{key}"
                    if slice_file_key in loaded:
                        self.slices[str(key)] = torch.from_numpy(loaded[slice_file_key])

            print("Cache loaded successfully.")

        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            self.process_raw()
            self.save_cache()


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed()

    # Train Dataset
    train_dataset = MoleculeDataset(
        root=Config.WORKING_DIR,
        metadata_file=Config.TRAIN_METADATA_PATH,
        structures_file=Config.STRUCTURES_PATH,
        mode="train",
        load_cached=load_cached_data,
    )

    # Validation Dataset
    val_dataset = MoleculeDataset(
        root=Config.WORKING_DIR,
        metadata_file=Config.VAL_METADATA_PATH,
        structures_file=Config.STRUCTURES_PATH,
        mode="val",
        load_cached=load_cached_data,
    )

    # Test Dataset
    test_dataset = MoleculeDataset(
        root=Config.WORKING_DIR,
        metadata_file=Config.TEST_METADATA_PATH,
        structures_file=Config.STRUCTURES_PATH,
        mode="test",
        load_cached=load_cached_data,
    )

    # Create DataLoaders
    # Using follow_batch=[] if needed, but standard collation works for this GCN

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
