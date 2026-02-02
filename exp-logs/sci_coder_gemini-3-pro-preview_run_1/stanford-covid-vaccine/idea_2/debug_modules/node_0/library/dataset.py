import torch
from torch_geometric.data import InMemoryDataset
from library.config import Config
from library.utils import load_data


class RNAGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for RNA Degradation Prediction.
    Wraps the data loading logic provided in library.utils to be compatible with PyG DataLoaders.
    """

    def __init__(
        self,
        split="train",
        limit=None,
        load_cached_data=True,
        root=None,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            split (str): The dataset split to load ('train', 'val', or 'test').
            limit (int, optional): If provided, limits the number of samples loaded (useful for debugging).
            load_cached_data (bool): Whether to attempt loading from the cache first.
            root (str, optional): Root directory for the dataset. Defaults to Config.WORKING_DIR.
            transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version.
        """
        self.split = split
        self.limit = limit
        self.load_cached_data = load_cached_data

        # Set default root to the working directory defined in Config
        if root is None:
            root = Config.WORKING_DIR

        super().__init__(root, transform, pre_transform)

        # Load the list of PyG Data objects using the utility function.
        # This function handles:
        # 1. Checking if a cached .pt file exists (if load_cached_data is True).
        # 2. If not, reading the metadata Parquet file.
        # 3. Processing rows into Graph objects.
        # 4. Saving the result to cache.
        data_list = load_data(split=self.split, load_cached_data=self.load_cached_data)

        # Apply dataset limit if requested
        if self.limit is not None:
            data_list = data_list[: self.limit]

        # Collate the list of Data objects into the internal storage format of InMemoryDataset
        self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        """
        Returns an empty list because raw file handling is managed by library.utils.load_data
        reading from the metadata directory directly.
        """
        return []

    @property
    def processed_file_names(self):
        """
        Returns an empty list because processed file caching is managed by library.utils.load_data
        using specific cache paths defined in Config.
        """
        return []

    def download(self):
        """
        No download step required. Data is expected to be in ./metadata.
        """
        pass

    def process(self):
        """
        No explicit process step required in this class.
        Processing logic is encapsulated in library.utils.load_data which is called in __init__.
        """
        pass
