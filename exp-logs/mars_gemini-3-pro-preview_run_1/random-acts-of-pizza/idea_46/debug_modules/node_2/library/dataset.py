import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.data_loader import get_processed_features
from library.config import Config


class PizzaDataset(Dataset):
    """
    Custom Dataset for the Orthogonal Skip-Gated MLP.
    Wraps pre-processed SBERT embeddings, history sequences, and metadata features.
    """

    def __init__(self, features_dict, labels=None):
        """
        Args:
            features_dict (dict): Dictionary containing numpy arrays for features.
                                  Expected keys: 'title', 'body', 'hist', 'mask',
                                  'cent', 'meta_dense', 'meta_skip'.
            labels (np.ndarray, optional): Binary target labels.
        """
        super().__init__()

        # Convert features to FloatTensor
        self.title_emb = torch.tensor(features_dict["title"], dtype=torch.float32)
        self.body_emb = torch.tensor(features_dict["body"], dtype=torch.float32)
        self.history_emb = torch.tensor(features_dict["hist"], dtype=torch.float32)
        self.history_mask = torch.tensor(features_dict["mask"], dtype=torch.float32)
        self.persona_centroid = torch.tensor(features_dict["cent"], dtype=torch.float32)
        self.metadata_dense = torch.tensor(
            features_dict["meta_dense"], dtype=torch.float32
        )
        self.metadata_skip = torch.tensor(
            features_dict["meta_skip"], dtype=torch.float32
        )

        # Handle labels
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        else:
            self.labels = None

        # Verify lengths
        assert len(self.title_emb) == len(
            self.metadata_dense
        ), "Feature length mismatch"
        if self.labels is not None:
            assert len(self.labels) == len(self.title_emb), "Label length mismatch"

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        """
        Returns a dictionary of tensors matching the signature of OrthogonalSkipGatedMLP.forward
        """
        sample = {
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "history_emb": self.history_emb[idx],
            "history_mask": self.history_mask[idx],
            "persona_centroid": self.persona_centroid[idx],
            "metadata_dense": self.metadata_dense[idx],
            "metadata_skip": self.metadata_skip[idx],
        }

        if self.labels is not None:
            sample["label"] = self.labels[idx]

        return sample


def create_dataloaders(batch_size=Config.MLP_BATCH_SIZE, load_cached_data=True):
    """
    Loads processed features and creates DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        load_cached_data (bool): Whether to load data from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load processed data using the library function
    data = get_processed_features(load_cached_data=load_cached_data)

    # Initialize Datasets
    train_dataset = PizzaDataset(data["mlp_train"], data["y_train"])
    val_dataset = PizzaDataset(data["mlp_val"], data["y_val"])
    test_dataset = PizzaDataset(data["mlp_test"], labels=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
