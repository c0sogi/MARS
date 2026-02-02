import torch
from torch.utils.data import Dataset
from library import data_utils


class RMSHDDataset(Dataset):
    """
    PyTorch Dataset for the RMS-HD Network.

    This dataset implements the data ingestion pipeline for Glioblastoma subtype prediction.
    It delegates the complex data processing (loading DICOMs, sorting by instance number,
    uniform high-density sampling, and global volumetric normalization) to the
    library.data_utils module.

    It strictly adheres to the caching mechanism provided by data_utils to ensure
    efficient training and reproducibility.
    """

    def __init__(self, df, subset_name="train", load_cached_data=True, transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing BraTS21ID and file paths.
            subset_name (str): Unique identifier for this dataset split (e.g., 'train', 'val', 'test').
                               This is used to name the cache files (e.g., cached_train_X.npy).
            load_cached_data (bool): If True, attempts to load pre-processed data from the
                                     working directory cache. If False, forces reprocessing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.subset_name = subset_name

        # Load the dataset using the provided utility.
        # This function handles the logic for:
        # 1. Checking if cache exists in ./working/idea_23/
        # 2. Loading from cache if available and requested.
        # 3. Processing from scratch using load_patient_volume if cache is missing.
        #    (This includes sorting, 32-slice sampling, and normalization).
        # 4. Saving the processed arrays to cache for future runs.
        self.X, self.y, self.ids = data_utils.load_dataset(
            metadata_df=df, cache_name=subset_name, load_cached_data=load_cached_data
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (torch.FloatTensor): The MRI volume tensor of shape (128, 224, 224).
            target (torch.FloatTensor): The target label (0.0 or 1.0). Returns -1.0 for test data.
        """
        # Retrieve the image volume from the loaded array
        # Shape: (128, 224, 224)
        image_np = self.X[idx]

        # Convert to PyTorch Tensor
        image = torch.from_numpy(image_np).float()

        # Apply optional transforms (if any)
        if self.transform:
            image = self.transform(image)

        # Retrieve the target label
        if self.y is not None:
            # Training/Validation set: return the actual label
            target_val = self.y[idx]
            target = torch.tensor(target_val, dtype=torch.float32)
        else:
            # Test set: return a dummy label
            target = torch.tensor(-1.0, dtype=torch.float32)

        return image, target

    def get_ids(self):
        """
        Returns the array of BraTS21IDs associated with the dataset.
        Useful for aligning predictions with patient IDs during submission generation.
        """
        return self.ids
