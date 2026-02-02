import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config, ATOM_MAP, TYPE_MAP
from library.utils import Standardizer


def preprocess_data(config: Config, split: str, load_cached_data: bool = True):
    """
    Preprocesses data into a flattened Structure-of-Arrays (SoA) format.
    Batches by molecule to enable efficient GNN processing.

    Args:
        config: Configuration object.
        split: 'train', 'val', or 'test'.
        load_cached_data: Whether to try loading from disk first.

    Returns:
        Dictionary containing numpy arrays for the dataset.
    """
    cache_dir = os.path.join(config.WORKING_DIR, "processed", split)
    os.makedirs(cache_dir, exist_ok=True)

    # Define filenames for cached arrays
    files = {
        "mol_names": "mol_names.npy",
        "mol_atom_ptr": "mol_atom_ptr.npy",
        "mol_atom_count": "mol_atom_count.npy",
        "atom_types": "atom_types.npy",
        "atom_coords": "atom_coords.npy",
        "mol_coupling_ptr": "mol_coupling_ptr.npy",
        "mol_coupling_count": "mol_coupling_count.npy",
        "coupling_atom_index_0": "coupling_atom_index_0.npy",
        "coupling_atom_index_1": "coupling_atom_index_1.npy",
        "coupling_type": "coupling_type.npy",
        "coupling_value": "coupling_value.npy",
        "coupling_id": "coupling_id.npy",
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(
            os.path.exists(os.path.join(cache_dir, f)) for f in files.values()
        )
        if all_exist:
            print(f"Loading cached {split} data from {cache_dir}...")
            data = {}
            for k, v in files.items():
                data[k] = np.load(os.path.join(cache_dir, v), allow_pickle=True)
            return data
        else:
            print(f"Cache miss for {split}. Processing from scratch...")

    # 2. Determine Metadata Path
    if split == "train":
        meta_path = config.TRAIN_METADATA
    elif split == "val":
        meta_path = config.VAL_METADATA
    elif split == "test":
        meta_path = config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    # 3. Load Metadata
    print(f"Loading metadata from {meta_path}...")
    df_meta = pd.read_csv(meta_path)

    # Debugging: Subsample
    if config.debug:
        print(f"DEBUG MODE: Subsampling {config.debug_samples} molecules...")
        unique_mols = df_meta["molecule_name"].unique()
        if len(unique_mols) > config.debug_samples:
            selected_mols = unique_mols[: config.debug_samples]
            df_meta = df_meta[df_meta["molecule_name"].isin(selected_mols)].copy()

    # 4. Fit Standardizer (only on training split)
    if split == "train":
        standardizer = Standardizer(config)
        standardizer.fit_or_load(df_meta, load_cached_data=load_cached_data)

    # 5. Load Structures
    print(f"Loading structures from {config.STRUCTURES_CSV}...")
    df_struct = pd.read_csv(config.STRUCTURES_CSV)

    # Filter structures to relevant molecules
    relevant_mols = df_meta["molecule_name"].unique()
    df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)].copy()

    # 6. Sort Data for Alignment
    # We must ensure that molecules appear in the exact same order in both arrays.
    print("Sorting and processing structures...")
    df_struct.sort_values(["molecule_name", "atom_index"], inplace=True)

    print("Sorting and processing metadata...")
    df_meta.sort_values(["molecule_name"], inplace=True)

    # 7. Process Atoms (Nodes)
    # Map atom types to integers
    df_struct["type_idx"] = df_struct["atom"].map(ATOM_MAP).astype(np.int8)

    # Extract arrays
    atom_types = df_struct["type_idx"].values
    atom_coords = df_struct[["x", "y", "z"]].values.astype(np.float32)

    # Compute molecule pointers
    # Group by molecule to get counts. Since we sorted, keys are sorted.
    struct_grp = df_struct.groupby("molecule_name", sort=False)
    mol_names_struct = np.array(list(struct_grp.groups.keys()))
    mol_atom_count = struct_grp.size().values.astype(np.int32)

    # Calculate pointers (cumsum)
    mol_atom_ptr = np.zeros(len(mol_atom_count), dtype=np.int32)
    mol_atom_ptr[1:] = np.cumsum(mol_atom_count)[:-1]

    # 8. Process Couplings (Edges/Targets)
    df_meta["type_idx"] = df_meta["type"].map(TYPE_MAP).astype(np.int8)

    coupling_atom_0 = df_meta["atom_index_0"].values.astype(np.int16)
    coupling_atom_1 = df_meta["atom_index_1"].values.astype(np.int16)
    coupling_type = df_meta["type_idx"].values
    coupling_id = df_meta["id"].values.astype(np.int32)

    if "scalar_coupling_constant" in df_meta.columns:
        coupling_value = df_meta["scalar_coupling_constant"].values.astype(np.float32)
    else:
        coupling_value = np.zeros(len(df_meta), dtype=np.float32)

    # Group by molecule
    meta_grp = df_meta.groupby("molecule_name", sort=False)
    mol_names_meta = np.array(list(meta_grp.groups.keys()))
    mol_coupling_count = meta_grp.size().values.astype(np.int32)

    mol_coupling_ptr = np.zeros(len(mol_coupling_count), dtype=np.int32)
    mol_coupling_ptr[1:] = np.cumsum(mol_coupling_count)[:-1]

    # 9. Verify Alignment
    if not np.array_equal(mol_names_struct, mol_names_meta):
        raise ValueError(
            "Molecule alignment failed between structures and metadata. Ensure data integrity."
        )

    # 10. Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(os.path.join(cache_dir, files["mol_names"]), mol_names_struct)
    np.save(os.path.join(cache_dir, files["mol_atom_ptr"]), mol_atom_ptr)
    np.save(os.path.join(cache_dir, files["mol_atom_count"]), mol_atom_count)
    np.save(os.path.join(cache_dir, files["atom_types"]), atom_types)
    np.save(os.path.join(cache_dir, files["atom_coords"]), atom_coords)

    np.save(os.path.join(cache_dir, files["mol_coupling_ptr"]), mol_coupling_ptr)
    np.save(os.path.join(cache_dir, files["mol_coupling_count"]), mol_coupling_count)
    np.save(os.path.join(cache_dir, files["coupling_atom_index_0"]), coupling_atom_0)
    np.save(os.path.join(cache_dir, files["coupling_atom_index_1"]), coupling_atom_1)
    np.save(os.path.join(cache_dir, files["coupling_type"]), coupling_type)
    np.save(os.path.join(cache_dir, files["coupling_value"]), coupling_value)
    np.save(os.path.join(cache_dir, files["coupling_id"]), coupling_id)

    data = {
        "mol_names": mol_names_struct,
        "mol_atom_ptr": mol_atom_ptr,
        "mol_atom_count": mol_atom_count,
        "atom_types": atom_types,
        "atom_coords": atom_coords,
        "mol_coupling_ptr": mol_coupling_ptr,
        "mol_coupling_count": mol_coupling_count,
        "coupling_atom_index_0": coupling_atom_0,
        "coupling_atom_index_1": coupling_atom_1,
        "coupling_type": coupling_type,
        "coupling_value": coupling_value,
        "coupling_id": coupling_id,
    }
    return data


class MoleculeDataset(Dataset):
    """
    Dataset that yields full molecular graphs and their associated coupling targets.
    Uses the preprocessed SoA data for efficient access.
    """

    def __init__(
        self, config: Config, split: str = "train", load_cached_data: bool = True
    ):
        self.config = config
        self.split = split
        self.data = preprocess_data(config, split, load_cached_data)
        self.num_molecules = len(self.data["mol_names"])

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        # Retrieve Atom (Node) Information
        a_ptr = self.data["mol_atom_ptr"][idx]
        a_cnt = self.data["mol_atom_count"][idx]

        atom_types = self.data["atom_types"][a_ptr : a_ptr + a_cnt]
        atom_coords = self.data["atom_coords"][a_ptr : a_ptr + a_cnt]

        # Retrieve Coupling (Edge/Target) Information
        c_ptr = self.data["mol_coupling_ptr"][idx]
        c_cnt = self.data["mol_coupling_count"][idx]

        c_atom0 = self.data["coupling_atom_index_0"][c_ptr : c_ptr + c_cnt]
        c_atom1 = self.data["coupling_atom_index_1"][c_ptr : c_ptr + c_cnt]
        c_type = self.data["coupling_type"][c_ptr : c_ptr + c_cnt]
        c_val = self.data["coupling_value"][c_ptr : c_ptr + c_cnt]
        c_id = self.data["coupling_id"][c_ptr : c_ptr + c_cnt]

        return {
            "atom_types": torch.from_numpy(atom_types).long(),
            "atom_coords": torch.from_numpy(atom_coords).float(),
            "coupling_atom_index_0": torch.from_numpy(c_atom0).long(),
            "coupling_atom_index_1": torch.from_numpy(c_atom1).long(),
            "coupling_type": torch.from_numpy(c_type).long(),
            "coupling_value": torch.from_numpy(c_val).float(),
            "coupling_id": torch.from_numpy(c_id).long(),
            "num_atoms": a_cnt,
            "num_couplings": c_cnt,
        }


def collate_molecules(batch):
    """
    Collates a list of molecule dictionaries into a single batch.

    Crucially, this function adjusts the atom indices in 'coupling_atom_index'
    to point to the correct atoms in the concatenated 'atom_types/coords' arrays.
    """
    # Initialize lists for concatenation
    atom_types_list = []
    atom_coords_list = []
    batch_index_list = []

    coupling_atom0_list = []
    coupling_atom1_list = []
    coupling_type_list = []
    coupling_value_list = []
    coupling_id_list = []

    atom_offset = 0

    for i, sample in enumerate(batch):
        num_atoms = sample["num_atoms"]

        # --- Process Nodes ---
        atom_types_list.append(sample["atom_types"])
        atom_coords_list.append(sample["atom_coords"])
        # Create a batch index vector (e.g., [0, 0, 0, 1, 1, ...])
        batch_index_list.append(torch.full((num_atoms,), i, dtype=torch.long))

        # --- Process Edges (Couplings) ---
        # Shift the local atom indices by the current cumulative atom offset
        # so they point to the correct global index in the batch
        coupling_atom0_list.append(sample["coupling_atom_index_0"] + atom_offset)
        coupling_atom1_list.append(sample["coupling_atom_index_1"] + atom_offset)

        coupling_type_list.append(sample["coupling_type"])
        coupling_value_list.append(sample["coupling_value"])
        coupling_id_list.append(sample["coupling_id"])

        # Update offset for the next molecule
        atom_offset += num_atoms

    # Concatenate everything
    batch_data = {
        "atom_types": torch.cat(atom_types_list, dim=0),
        "atom_coords": torch.cat(atom_coords_list, dim=0),
        "batch_index": torch.cat(batch_index_list, dim=0),
        "coupling_atom_index_0": torch.cat(coupling_atom0_list, dim=0),
        "coupling_atom_index_1": torch.cat(coupling_atom1_list, dim=0),
        "coupling_type": torch.cat(coupling_type_list, dim=0),
        "coupling_value": torch.cat(coupling_value_list, dim=0),
        "coupling_id": torch.cat(coupling_id_list, dim=0),
        "num_graphs": len(batch),
    }

    return batch_data
