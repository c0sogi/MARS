import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from library.config import (
    IMAGES_DIR,
    CACHE_IMAGE_EMBEDDINGS,
    CACHE_IMAGE_ID_MAP,
    SEED,
    N_CPUS,
    INPUT_DIR,
)

# Set fixed seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class HNMImageDataset(Dataset):
    """
    Custom Dataset for H&M images.
    """

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.base_path = INPUT_DIR

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        # Full path construction
        full_path = self.base_path / rel_path

        try:
            image = Image.open(full_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for corrupted or missing images despite pre-check
            # Return a black image
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        return image


class ImageEmbedder:
    """
    Wrapper around a pre-trained ResNet model for feature extraction.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load pre-trained ResNet18
        # We use the modern weights parameter if available, else pretrained=True
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.model = models.resnet18(pretrained=True)

        # Remove the fully connected layer (classification head)
        # ResNet architecture: ... -> avgpool -> fc
        # We want the output of avgpool.
        # Replacing fc with Identity effectively returns the flattened pool output if we handle shapes correctly,
        # but standard practice is often to just use the backbone.
        # Here we replace the fc layer with an Identity layer.
        self.model.fc = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_transform(self):
        return self.transform

    def extract(self, dataloader):
        embeddings = []
        with torch.no_grad():
            for imgs in dataloader:
                imgs = imgs.to(self.device)
                # Forward pass
                # ResNet18 avgpool output is (Batch, 512, 1, 1), flattened by Identity to (Batch, 512)
                feats = self.model(imgs)
                embeddings.append(feats.cpu().numpy())

        if len(embeddings) > 0:
            return np.vstack(embeddings)
        return np.array([])


def extract_all_embeddings(
    article_ids, image_paths, load_cached_data=True, batch_size=256
):
    """
    Extracts visual embeddings for a list of articles.

    Parameters
    ----------
    article_ids : list or np.array
        List of article IDs corresponding to the paths.
    image_paths : list or np.array
        List of relative image paths (e.g., 'images/010/0108775015.jpg').
    load_cached_data : bool
        Whether to load from disk if available.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    dict
        Mapping {article_id (str): embedding (np.array)}.
    """

    # 1. Check Cache
    if load_cached_data:
        if CACHE_IMAGE_EMBEDDINGS.exists() and CACHE_IMAGE_ID_MAP.exists():
            print(f"Loading cached image embeddings from {CACHE_IMAGE_EMBEDDINGS}...")
            try:
                embeddings = np.load(CACHE_IMAGE_EMBEDDINGS)
                id_map = np.load(CACHE_IMAGE_ID_MAP, allow_pickle=True)

                # Reconstruct dictionary
                # Ensure id_map are strings
                id_map_str = id_map.astype(str)
                embedding_dict = dict(zip(id_map_str, embeddings))
                return embedding_dict
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print("Cached embeddings not found. Computing from scratch...")

    # 2. Filter Valid Images
    # We only process images that actually exist to avoid crashing the DataLoader
    print("Verifying image paths...")
    valid_ids = []
    valid_paths = []

    # Convert inputs to lists if they aren't already
    article_ids_list = list(article_ids)
    image_paths_list = list(image_paths)

    # Check existence
    # Note: Checking file existence one by one can be slow, but necessary for robustness.
    # We assume INPUT_DIR is the base.
    base_path = INPUT_DIR

    for aid, path in zip(article_ids_list, image_paths_list):
        if (base_path / path).exists():
            valid_ids.append(str(aid))
            valid_paths.append(path)

    print(f"Found {len(valid_paths)} valid images out of {len(image_paths)} requested.")

    if len(valid_paths) == 0:
        print("No valid images found. Returning empty embedding dictionary.")
        return {}

    # 3. Setup Model and Data
    embedder = ImageEmbedder()
    dataset = HNMImageDataset(valid_paths, transform=embedder.get_transform())

    # Use num_workers for faster loading, but ensure it doesn't exceed CPU limits
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, N_CPUS),
        pin_memory=True,
    )

    # 4. Extract
    print(f"Starting embedding extraction on {embedder.device}...")
    embeddings = embedder.extract(loader)

    # 5. Save to Cache
    print(f"Saving embeddings to {CACHE_IMAGE_EMBEDDINGS}...")
    valid_ids_np = np.array(valid_ids)

    # Ensure directory exists
    os.makedirs(CACHE_IMAGE_EMBEDDINGS.parent, exist_ok=True)

    np.save(CACHE_IMAGE_EMBEDDINGS, embeddings)
    np.save(CACHE_IMAGE_ID_MAP, valid_ids_np)

    # 6. Return Dictionary
    embedding_dict = dict(zip(valid_ids, embeddings))
    return embedding_dict
