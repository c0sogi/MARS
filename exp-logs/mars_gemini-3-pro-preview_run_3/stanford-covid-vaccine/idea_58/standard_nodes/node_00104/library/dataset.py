import torch
from torch.utils.data import Dataset
from library.data_utils import load_dataset


class RNADataset(Dataset):
    """
    PyTorch Dataset for the RNA Degradation Prediction task.

    This class wraps the data loading and processing logic defined in library.data_utils.
    It handles the conversion of numpy arrays to PyTorch tensors suitable for the
    High-Capacity Stabilized Decoupled BiGRU (HCSD-BiGRU) model.
    """

    def __init__(self, mode="train", load_cached_data=True, max_samples=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load pre-processed data from .npz cache.
                                     If False or cache missing, re-processes data and saves cache.
            max_samples (int, optional): Limit the dataset size for debugging purposes.
        """
        self.mode = mode

        # Load data using the provided utility function which handles caching and processing
        self.data_dict = load_dataset(
            mode=mode, load_cached_data=load_cached_data, max_samples=max_samples
        )

        # Unpack necessary arrays for easier access
        self.ids = self.data_dict["ids"]
        self.features = self.data_dict["features"]
        self.bpp_indices = self.data_dict["bpp_indices"]
        self.bpp_mask = self.data_dict["bpp_mask"]

        # Targets are only available for train and val sets
        self.targets = self.data_dict.get("targets", None)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'features': Tensor (Seq_Len, 14) - Float32
                - 'bpp_indices': Tensor (Seq_Len,) - Long (for gathering)
                - 'bpp_mask': Tensor (Seq_Len,) - Float32
                - 'ids': str - Sample ID
                - 'targets': Tensor (Seq_Scored, 5) - Float32 (Only if available)
        """
        # Convert numpy arrays to PyTorch tensors
        # Features: (107, 14)
        features_tensor = torch.from_numpy(self.features[idx]).float()

        # BPP Indices: (107,) - Must be Long for indexing/gather
        bpp_indices_tensor = torch.from_numpy(self.bpp_indices[idx]).long()

        # BPP Mask: (107,)
        bpp_mask_tensor = torch.from_numpy(self.bpp_mask[idx]).float()

        sample = {
            "features": features_tensor,
            "bpp_indices": bpp_indices_tensor,
            "bpp_mask": bpp_mask_tensor,
            "ids": str(self.ids[idx]),
        }

        # Add targets if they exist (Train/Val modes)
        if self.targets is not None:
            # Targets: (68, 5)
            targets_tensor = torch.from_numpy(self.targets[idx]).float()
            sample["targets"] = targets_tensor

        return sample
