import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import Standardizer


def process_and_cache(
    metadata_path, cache_dir, structures_df, mode="train", load_cached_data=True
):
    """
    Processes the metadata and structures into a molecule-centric format.
    Caches the results as numpy object arrays for efficient loading.
    """
    # Define cache file paths
    cache_files = {
        "mol_names": os.path.join(cache_dir, "mol_names.npy"),
        "atom_types": os.path.join(cache_dir, "atom_types.npy"),
        "atom_coords": os.path.join(cache_dir, "atom_coords.npy"),
        "coupling_pairs": os.path.join(cache_dir, "coupling_pairs.npy"),
        "coupling_types": os.path.join(cache_dir, "coupling_types.npy"),
        "coupling_values": os.path.join(cache_dir, "coupling_values.npy"),
        "coupling_ids": os.path.join(cache_dir, "coupling_ids.npy"),
    }

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        print(f"Loading cached {mode} data from {cache_dir}...")
        data = {}
        for key, path in cache_files.items():
            data[key] = np.load(path, allow_pickle=True)
        return data

    print(f"Processing {mode} data from scratch...")

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Debug mode: sample subset
    if Config.DEBUG:
        print(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} molecules...")
        unique_mols = df["molecule_name"].unique()
        if len(unique_mols) > Config.DEBUG_SAMPLE_SIZE:
            sampled_mols = np.random.choice(
                unique_mols, Config.DEBUG_SAMPLE_SIZE, replace=False
            )
            df = df[df["molecule_name"].isin(sampled_mols)].copy()

    # Map coupling types to integers
    if "type" in df.columns:
        df["type"] = df["type"].astype(str)
        df["type_idx"] = df["type"].map(Config.COUPLING_TYPE_MAP)

    # Sort by molecule_name to ensure deterministic order
    df = df.sort_values("molecule_name")

    # Group couplings by molecule
    mol_groups = df.groupby("molecule_name")

    # Prepare structures
    # Filter structures to only include molecules in this split
    relevant_mols = df["molecule_name"].unique()
    struct_df = structures_df[structures_df["molecule_name"].isin(relevant_mols)].copy()

    # Map atom types
    struct_df["atom_idx"] = struct_df["atom"].map(Config.ATOM_MAP)

    # Create a dictionary for structures for O(1) access
    # We iterate the groupby object once
    struct_dict = {}
    for name, group in struct_df.groupby("molecule_name"):
        types = group["atom_idx"].values.astype(np.int64)
        coords = group[["x", "y", "z"]].values.astype(np.float32)
        struct_dict[name] = (types, coords)

    # Containers
    mol_names_list = []
    atom_types_list = []
    atom_coords_list = []
    coupling_pairs_list = []
    coupling_types_list = []
    coupling_values_list = []
    coupling_ids_list = []

    # Iterate over molecules
    for name, group in mol_groups:
        if name in struct_dict:
            atom_types, atom_coords = struct_dict[name]
        else:
            # Fallback: Load from XYZ file if missing from structures.csv
            try:
                rel_path = group.iloc[0]["structure_path"]
                full_path = os.path.join(Config.INPUT_DIR, rel_path)

                if not os.path.exists(full_path):
                    continue

                with open(full_path, "r") as f:
                    lines = f.readlines()

                # Parse XYZ
                a_types = []
                a_coords = []
                for line in lines[2:]:
                    parts = line.split()
                    if not parts:
                        continue
                    symbol = parts[0]
                    if symbol in Config.ATOM_MAP:
                        a_types.append(Config.ATOM_MAP[symbol])
                        a_coords.append([float(x) for x in parts[1:4]])

                atom_types = np.array(a_types, dtype=np.int64)
                atom_coords = np.array(a_coords, dtype=np.float32)

            except Exception:
                continue

        # Coupling data
        # pairs: (N_couplings, 2)
        pairs = group[["atom_index_0", "atom_index_1"]].values.astype(np.int64)

        # types: (N_couplings,)
        c_types = group["type_idx"].values.astype(np.int64)

        # ids: (N_couplings,)
        c_ids = group["id"].values.astype(np.int64)

        # values: (N_couplings,) - only if exists (Train/Val)
        if "scalar_coupling_constant" in group.columns:
            c_values = group["scalar_coupling_constant"].values.astype(np.float32)
        else:
            c_values = np.zeros(len(group), dtype=np.float32)

        mol_names_list.append(name)
        atom_types_list.append(atom_types)
        atom_coords_list.append(atom_coords)
        coupling_pairs_list.append(pairs)
        coupling_types_list.append(c_types)
        coupling_values_list.append(c_values)
        coupling_ids_list.append(c_ids)

    # Convert to numpy object arrays
    data = {
        "mol_names": np.array(mol_names_list, dtype=object),
        "atom_types": np.array(atom_types_list, dtype=object),
        "atom_coords": np.array(atom_coords_list, dtype=object),
        "coupling_pairs": np.array(coupling_pairs_list, dtype=object),
        "coupling_types": np.array(coupling_types_list, dtype=object),
        "coupling_values": np.array(coupling_values_list, dtype=object),
        "coupling_ids": np.array(coupling_ids_list, dtype=object),
    }

    # Save to cache
    print(f"Saving processed {mode} data to {cache_dir}...")
    for key, arr in data.items():
        np.save(cache_files[key], arr)

    return data


class MoleculeDataset(Dataset):
    def __init__(self, data):
        self.mol_names = data["mol_names"]
        self.atom_types = data["atom_types"]
        self.atom_coords = data["atom_coords"]
        self.coupling_pairs = data["coupling_pairs"]
        self.coupling_types = data["coupling_types"]
        self.coupling_values = data["coupling_values"]
        self.coupling_ids = data["coupling_ids"]

    def __len__(self):
        return len(self.mol_names)

    def __getitem__(self, idx):
        return {
            "mol_name": self.mol_names[idx],
            "atom_types": torch.tensor(self.atom_types[idx], dtype=torch.long),
            "atom_coords": torch.tensor(self.atom_coords[idx], dtype=torch.float32),
            "coupling_pairs": torch.tensor(self.coupling_pairs[idx], dtype=torch.long),
            "coupling_types": torch.tensor(self.coupling_types[idx], dtype=torch.long),
            "coupling_values": torch.tensor(
                self.coupling_values[idx], dtype=torch.float32
            ),
            "coupling_ids": torch.tensor(self.coupling_ids[idx], dtype=torch.long),
            "num_atoms": len(self.atom_types[idx]),
        }


def collate_molecules(batch):
    """
    Collates a list of molecule dictionaries into a single batch.
    Handles offsetting of node indices for coupling pairs.
    """
    batch_atom_types = []
    batch_atom_coords = []
    batch_batch_index = []  # Maps atoms to molecule index in batch

    batch_coupling_pairs = []
    batch_coupling_types = []
    batch_coupling_values = []
    batch_coupling_ids = []

    batch_mol_names = []

    atom_offset = 0

    for i, item in enumerate(batch):
        # Nodes
        num_atoms = item["num_atoms"]
        batch_atom_types.append(item["atom_types"])
        batch_atom_coords.append(item["atom_coords"])
        batch_batch_index.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Couplings
        # Offset the node indices in pairs so they point to the correct atoms in the batched list
        pairs = item["coupling_pairs"] + atom_offset
        batch_coupling_pairs.append(pairs)

        batch_coupling_types.append(item["coupling_types"])
        batch_coupling_values.append(item["coupling_values"])
        batch_coupling_ids.append(item["coupling_ids"])

        batch_mol_names.append(item["mol_name"])

        atom_offset += num_atoms

    return {
        "atom_types": torch.cat(batch_atom_types, dim=0),
        "atom_coords": torch.cat(batch_atom_coords, dim=0),
        "batch_index": torch.cat(batch_batch_index, dim=0),
        "coupling_pairs": torch.cat(batch_coupling_pairs, dim=0),
        "coupling_types": torch.cat(batch_coupling_types, dim=0),
        "coupling_values": torch.cat(batch_coupling_values, dim=0),
        "coupling_ids": torch.cat(batch_coupling_ids, dim=0),
        "mol_names": batch_mol_names,
        "num_graphs": len(batch),
    }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get dataloaders.
    """
    # Load structures once
    print("Loading structures.csv...")
    structures_df = pd.read_csv(Config.STRUCTURES_CSV)

    # Process Train
    train_data = process_and_cache(
        Config.TRAIN_META_PATH,
        Config.TRAIN_CACHE_DIR,
        structures_df,
        mode="train",
        load_cached_data=load_cached_data,
    )

    # Process Val
    val_data = process_and_cache(
        Config.VAL_META_PATH,
        Config.VAL_CACHE_DIR,
        structures_df,
        mode="val",
        load_cached_data=load_cached_data,
    )

    # Process Test
    test_data = process_and_cache(
        Config.TEST_META_PATH,
        Config.TEST_CACHE_DIR,
        structures_df,
        mode="test",
        load_cached_data=load_cached_data,
    )

    # Create Datasets
    train_dataset = MoleculeDataset(train_data)
    val_dataset = MoleculeDataset(val_data)
    test_dataset = MoleculeDataset(test_data)

    # Fit Standardizer
    standardizer = Standardizer()

    # Check if stats are already cached to avoid reconstructing the DF
    if not (load_cached_data and os.path.exists(Config.STATS_PATH)):
        print("Constructing DataFrame for Standardizer fitting...")
        # Flatten arrays for DF construction
        all_types = np.concatenate(train_data["coupling_types"])
        all_values = np.concatenate(train_data["coupling_values"])

        std_df = pd.DataFrame(
            {"type": all_types, "scalar_coupling_constant": all_values}
        )
        standardizer.fit(std_df, load_cached_data=load_cached_data)
    else:
        # If cached, we pass None and it will load from disk
        standardizer.fit(None, load_cached_data=load_cached_data)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, standardizer
