import torch
import numpy as np
from torch.utils.data import Dataset
from library.data_manager import get_clean_data
from library.feature_engineers import MetadataExtractor, TextProcessor


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request task.
    Handles Request SBERT embeddings, History SBERT sequences, Metadata, and Labels.
    """

    def __init__(self, request_emb, history_seq, metadata, labels=None):
        """
        Args:
            request_emb (np.ndarray): (N, 384) SBERT embeddings of the request.
            history_seq (np.ndarray): (N, 20, 384) SBERT embeddings of subreddit history.
            metadata (np.ndarray): (N, F) Scaled numerical metadata.
            labels (np.ndarray, optional): (N,) Binary labels.
        """
        self.request_emb = torch.tensor(request_emb, dtype=torch.float32)
        self.history_seq = torch.tensor(history_seq, dtype=torch.float32)
        self.metadata = torch.tensor(metadata, dtype=torch.float32)

        # Create attention mask for history
        # History is padded with zeros. If the sum of the embedding vector is 0, it's padding.
        # Mask: 1 for valid token, 0 for padding.
        # Shape: (N, 20)
        history_sums = np.abs(history_seq).sum(axis=-1)
        self.history_mask = torch.tensor(
            (history_sums > 1e-6).astype(np.float32), dtype=torch.float32
        )

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.request_emb)

    def __getitem__(self, idx):
        item = {
            "request_emb": self.request_emb[idx],
            "history_seq": self.history_seq[idx],
            "history_mask": self.history_mask[idx],
            "metadata": self.metadata[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


def get_pizza_datasets(load_cached_data=True, debug_mode=False, debug_size=50):
    """
    Orchestrates the loading of raw data, feature extraction, and creation of PyTorch Datasets.

    Args:
        load_cached_data (bool): Whether to use cached intermediate features.
        debug_mode (bool): Whether to use a small subset of data.
        debug_size (int): Number of samples in debug mode.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # 1. Load Cleaned Dataframes
    df_train, df_val, df_test = get_clean_data(
        load_cached_data=load_cached_data, debug_mode=debug_mode, debug_size=debug_size
    )

    # 2. Extract Metadata Features (Scaled)
    meta_extractor = MetadataExtractor()
    # Compute raw features
    train_meta_raw, val_meta_raw, test_meta_raw = meta_extractor.process(
        df_train, df_val, df_test, load_cached_data=load_cached_data
    )
    # Scale features (Arcsinh + StandardScaler)
    X_meta_train, X_meta_val, X_meta_test = meta_extractor.get_scaled_features(
        train_meta_raw, val_meta_raw, test_meta_raw
    )

    # 3. Extract Text Features (SBERT)
    text_processor = TextProcessor()

    # Request Embeddings (N, 384)
    X_req_train, X_req_val, X_req_test = text_processor.process_sbert_request(
        df_train, df_val, df_test, load_cached_data=load_cached_data
    )

    # History Embeddings (N, 20, 384)
    X_hist_train, X_hist_val, X_hist_test = text_processor.process_sbert_history(
        df_train, df_val, df_test, load_cached_data=load_cached_data
    )

    # 4. Extract Targets
    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values
    # Test set has no targets for prediction

    # 5. Create Datasets
    train_dataset = PizzaDataset(
        request_emb=X_req_train,
        history_seq=X_hist_train,
        metadata=X_meta_train,
        labels=y_train,
    )

    val_dataset = PizzaDataset(
        request_emb=X_req_val, history_seq=X_hist_val, metadata=X_meta_val, labels=y_val
    )

    test_dataset = PizzaDataset(
        request_emb=X_req_test,
        history_seq=X_hist_test,
        metadata=X_meta_test,
        labels=None,
    )

    return train_dataset, val_dataset, test_dataset
