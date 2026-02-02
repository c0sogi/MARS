import torch
from torch.utils.data import Dataset
from library.data_utils import load_and_preprocess_data


class MaterialsDataset(Dataset):
    """
    PyTorch Dataset for materials data.

    This dataset handles the loading of global features (lattice parameters, composition, etc.)
    and atomic features (point clouds representing atoms in the unit cell).
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing processed data arrays.
                              Expected keys: 'ids', 'global_features', 'atomic_features', 'targets'
        """
        self.ids = data_dict["ids"]
        # Convert global features to float32 tensor immediately
        self.global_features = torch.tensor(
            data_dict["global_features"], dtype=torch.float32
        )
        # Atomic features are kept as a list/array of arrays until __getitem__ to optimize memory usage
        # Each element is a (N_atoms, Feature_Dim) numpy array
        self.atomic_features = data_dict["atomic_features"]

        # Handle targets if they exist (Train/Val sets)
        if data_dict["targets"] is not None:
            self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a sample dictionary containing:
            - id: Sample identifier
            - global_features: Tensor of shape (Global_Dim,)
            - atomic_features: Tensor of shape (N_atoms, Atomic_Dim)
            - targets: Tensor of shape (Num_Targets,) if available
        """
        # Convert specific atomic feature array to tensor on demand
        atom_feats = torch.tensor(self.atomic_features[idx], dtype=torch.float32)

        sample = {
            "id": self.ids[idx],
            "global_features": self.global_features[idx],
            "atomic_features": atom_feats,
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        return sample


def collate_fn(batch):
    """
    Custom collate function to handle variable-size point clouds (atomic features).

    Instead of padding, this function concatenates all atomic features into a single
    large tensor and creates a 'batch_indices' tensor mapping each atom to its
    corresponding sample in the batch. This is efficient for scatter operations.

    Args:
        batch (list): List of sample dictionaries from MaterialsDataset.__getitem__

    Returns:
        dict: Batched data dictionary
    """
    # 1. Global Features: Simple stack -> (Batch_Size, Global_Dim)
    global_features = torch.stack([item["global_features"] for item in batch])

    # 2. IDs: List of identifiers
    ids = [item["id"] for item in batch]

    # 3. Atomic Features: Concatenate -> (Total_Atoms_In_Batch, Atomic_Dim)
    atomic_features_list = [item["atomic_features"] for item in batch]
    atomic_features = torch.cat(atomic_features_list, dim=0)

    # Create batch indices vector: [0, 0, ..., 1, 1, ..., B-1, B-1]
    # Shape: (Total_Atoms_In_Batch,)
    batch_indices_list = []
    for i, feats in enumerate(atomic_features_list):
        n_atoms = feats.shape[0]
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))
    batch_indices = torch.cat(batch_indices_list, dim=0)

    # 4. Targets: Stack if present -> (Batch_Size, Num_Targets)
    targets = None
    if "targets" in batch[0]:
        targets = torch.stack([item["targets"] for item in batch])

    return {
        "ids": ids,
        "global_features": global_features,
        "atomic_features": atomic_features,
        "batch_indices": batch_indices,
        "targets": targets,
    }


def get_datasets(debug=False, max_samples=None):
    """
    Loads processed data and returns Train, Val, and Test datasets.

    Uses library.data_utils.load_and_preprocess_data which handles:
    - Reading metadata CSVs
    - Extracting features from XYZ files
    - Computing derived features
    - Caching processed data to disk
    - Normalizing features

    Args:
        debug (bool): If True, limits dataset size for debugging purposes.
        max_samples (int): Specific number of samples to use if debug is True.
                           Defaults to 100 if not specified.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load and preprocess data using the provided utility
    train_data, val_data, test_data = load_and_preprocess_data()

    # Handle debugging/subsampling
    if debug:
        if max_samples is None:
            max_samples = 100

        print(f"Debug mode: Limiting datasets to {max_samples} samples.")

        for d in [train_data, val_data, test_data]:
            current_len = len(d["ids"])
            limit = min(current_len, max_samples)

            d["ids"] = d["ids"][:limit]
            d["global_features"] = d["global_features"][:limit]
            d["atomic_features"] = d["atomic_features"][:limit]
            if d["targets"] is not None:
                d["targets"] = d["targets"][:limit]

    # Initialize Dataset objects
    train_dataset = MaterialsDataset(train_data)
    val_dataset = MaterialsDataset(val_data)
    test_dataset = MaterialsDataset(test_data)

    return train_dataset, val_dataset, test_dataset
