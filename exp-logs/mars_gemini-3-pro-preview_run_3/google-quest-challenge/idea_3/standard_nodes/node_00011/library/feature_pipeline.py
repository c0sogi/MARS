import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import PathConfig, ModelConfig, TrainConfig
from library.model import SegmentAwareCrossEncoder
from library.dataset import load_data, QuestDataset
from library.utils import seed_everything


def extract_features(model, dataloader, device):
    """
    Runs inference on the dataloader using the model to extract features.

    Args:
        model: The SegmentAwareCrossEncoder model.
        dataloader: DataLoader containing the data.
        device: Torch device.

    Returns:
        features: Numpy array of shape (n_samples, feature_dim)
        targets: Numpy array of shape (n_samples, n_targets) or None if no labels.
    """
    model.eval()
    features_list = []
    targets_list = []

    # We use a larger batch size for inference, but we iterate through the provided dataloader
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_mask = batch["a_mask"].to(device)

            # Forward pass with Mixed Precision for speed
            with torch.amp.autocast(
                device_type="cuda", enabled=(device.type == "cuda")
            ):
                # The model returns (logits, features)
                _, features = model(input_ids, attention_mask, q_mask, a_mask)

            features_list.append(features.cpu().numpy())

            if "labels" in batch:
                targets_list.append(batch["labels"].cpu().numpy())

    # Concatenate all batches
    all_features = np.concatenate(features_list, axis=0)

    if len(targets_list) > 0:
        all_targets = np.concatenate(targets_list, axis=0)
    else:
        all_targets = None

    return all_features, all_targets


def run_feature_extraction(load_cached_data=True):
    """
    Main driver to load data, load model, extract features, and cache them.

    Args:
        load_cached_data (bool): Whether to try loading from disk first.

    Returns:
        train_features, train_targets, val_features, val_targets, test_features
    """
    # Define cache paths
    train_feat_path = PathConfig.TRAIN_FEATURES_CACHE
    train_targ_path = PathConfig.TRAIN_TARGETS_CACHE
    val_feat_path = PathConfig.VAL_FEATURES_CACHE
    val_targ_path = PathConfig.VAL_TARGETS_CACHE
    test_feat_path = PathConfig.TEST_FEATURES_CACHE

    # 1. Check Cache
    if load_cached_data:
        if (
            os.path.exists(train_feat_path)
            and os.path.exists(train_targ_path)
            and os.path.exists(val_feat_path)
            and os.path.exists(val_targ_path)
            and os.path.exists(test_feat_path)
        ):
            print("Loading extracted features from cache...")
            train_features = np.load(train_feat_path)
            train_targets = np.load(train_targ_path)
            val_features = np.load(val_feat_path)
            val_targets = np.load(val_targ_path)
            test_features = np.load(test_feat_path)

            return (
                train_features,
                train_targets,
                val_features,
                val_targets,
                test_features,
            )

    # 2. Setup for Extraction
    print("Starting feature extraction process...")
    seed_everything(TrainConfig.seed)
    device = torch.device(TrainConfig.device)

    # Load DataFrames
    # load_data handles its own caching of the parquet files
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.model_name)

    # Create Datasets
    train_dataset = QuestDataset(
        train_df, tokenizer, max_len=ModelConfig.max_len, mode="train"
    )
    val_dataset = QuestDataset(
        val_df, tokenizer, max_len=ModelConfig.max_len, mode="train"
    )
    test_dataset = QuestDataset(
        test_df, tokenizer, max_len=ModelConfig.max_len, mode="test"
    )

    # Create Dataloaders
    # Use larger batch size for inference (4x training batch size is usually safe)
    inference_batch_size = TrainConfig.batch_size * 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    # Load Model
    print(f"Loading model weights from {PathConfig.MODEL_SAVE_PATH}...")
    model = SegmentAwareCrossEncoder()

    if os.path.exists(PathConfig.MODEL_SAVE_PATH):
        state_dict = torch.load(PathConfig.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model weights not found at {PathConfig.MODEL_SAVE_PATH}. "
            "Using random initialization (results will be meaningless)."
        )

    model.to(device)

    # 3. Extract Features
    print("Extracting features for training set...")
    train_features, train_targets = extract_features(model, train_loader, device)

    print("Extracting features for validation set...")
    val_features, val_targets = extract_features(model, val_loader, device)

    print("Extracting features for test set...")
    test_features, _ = extract_features(model, test_loader, device)  # No targets

    # 4. Save to Cache
    print(f"Saving features to {PathConfig.WORKING_DIR}...")
    os.makedirs(PathConfig.WORKING_DIR, exist_ok=True)

    np.save(train_feat_path, train_features)
    np.save(train_targ_path, train_targets)
    np.save(val_feat_path, val_features)
    np.save(val_targ_path, val_targets)
    np.save(test_feat_path, test_features)

    return train_features, train_targets, val_features, val_targets, test_features
