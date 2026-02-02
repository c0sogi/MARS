import os
import torch
import timm
import numpy as np
import pandas as pd
from library.config import DEVICE, MODEL_NAME


class CNNBackbone(torch.nn.Module):
    """
    Wraps a pre-trained MobileNetV3 model from timm as a frozen feature extractor.
    """

    def __init__(self):
        super().__init__()
        # Create the model using timm.
        # num_classes=0 removes the final classification layer.
        # global_pool='avg' ensures the output is a 1D feature vector per image.
        try:
            self.model = timm.create_model(
                MODEL_NAME, pretrained=True, num_classes=0, global_pool="avg"
            )
        except Exception:
            # Fallback in case the config name differs slightly from timm's registry
            # but matches the prompt's description.
            self.model = timm.create_model(
                "mobilenetv3_large_100",
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )

        self.model.to(DEVICE)
        self.model.eval()

        # Freeze all parameters to ensure no training happens on the backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass to extract features.
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
        Returns:
            torch.Tensor: Feature vectors (B, Feature_Dim)
        """
        return self.model(x)


def extract_features(model, dataloader):
    """
    Iterates over the dataloader to extract image features and aggregate them
    with tabular features and targets.

    Args:
        model (CNNBackbone): The frozen backbone model.
        dataloader (DataLoader): DataLoader yielding (image, tabular, target).

    Returns:
        tuple: (img_features, tab_features, targets) as numpy arrays.
    """
    model.eval()

    img_feats_list = []
    tab_feats_list = []
    targets_list = []

    # Use torch.no_grad() for memory efficiency and speed during inference
    with torch.no_grad():
        for images, tabular, targets in dataloader:
            images = images.to(DEVICE)

            # Extract features from the backbone
            # Shape: (Batch_Size, Feature_Dim)
            features = model(images)

            # Move to CPU and convert to numpy
            img_feats_list.append(features.cpu().numpy())
            tab_feats_list.append(tabular.numpy())
            targets_list.append(targets.numpy())

    # Concatenate all batches into single arrays
    if len(img_feats_list) > 0:
        img_features = np.concatenate(img_feats_list, axis=0)
        tab_features = np.concatenate(tab_feats_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
    else:
        img_features = np.array([])
        tab_features = np.array([])
        targets = np.array([])

    return img_features, tab_features, targets


def process_and_cache_features(
    dataloader, cache_path, load_cached_data=True, model=None
):
    """
    Manages the extraction and caching of features.

    If the cache file exists and load_cached_data is True, it loads the data.
    Otherwise, it runs the feature extraction pipeline and saves the result to disk.

    Args:
        dataloader (DataLoader): The data source.
        cache_path (str): Path where the Parquet file is/will be stored.
        load_cached_data (bool): Flag to enable loading from cache.
        model (CNNBackbone, optional): Existing model instance. If None, one is created.

    Returns:
        tuple: (X, y)
            X (np.ndarray): Combined feature matrix (Image + Tabular).
            y (np.ndarray): Target vector.
    """
    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            y = df["target"].values
            X = df.drop(columns=["target"]).values
            return X, y
        except Exception as e:
            print(f"Error loading cache ({e}). Proceeding to re-extraction.")

    # 2. Perform Feature Extraction
    print(f"Extracting features for {cache_path}...")

    # Instantiate model if not provided
    if model is None:
        model = CNNBackbone()

    img_feats, tab_feats, targets = extract_features(model, dataloader)

    # 3. Combine Image and Tabular Features
    # X becomes [Image_Embedding_0 ... Image_Embedding_N, Tabular_0 ... Tabular_M]
    X = np.hstack([img_feats, tab_feats])
    y = targets

    # 4. Save to Parquet Cache
    # Generate column names for clarity in the dataframe
    n_img = img_feats.shape[1]
    n_tab = tab_feats.shape[1]

    cols = [f"img_{i}" for i in range(n_img)] + [f"tab_{i}" for i in range(n_tab)]

    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    # Ensure output directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    df.to_parquet(cache_path, index=False)
    print(f"Features successfully saved to {cache_path}")

    return X, y
