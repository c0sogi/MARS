import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from library import config, data_loader

# Set fixed seeds for reproducibility
SEED = config.SEED
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class HNMImageDataset(Dataset):
    """
    Custom Dataset for loading H&M images.
    """

    def __init__(self, article_ids, image_paths, transform=None):
        """
        Args:
            article_ids (list or np.array): List of article IDs.
            image_paths (list or np.array): List of corresponding image paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.article_ids = article_ids
        self.image_paths = image_paths
        self.transform = transform
        self.input_dir = config.INPUT_DIR

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        article_id = self.article_ids[idx]

        # Full path construction
        full_path = self.input_dir / img_path

        try:
            # Open image and convert to RGB
            image = Image.open(full_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, article_id, True
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            # Return a placeholder if image is corrupt or missing
            # In the collate_fn or loop, we will filter these out based on the success flag
            # Create a dummy tensor of correct shape (3, 224, 224)
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            return dummy_image, article_id, False


class VisualEncoder:
    """
    Wrapper for the visual encoding model.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"VisualEncoder using device: {self.device}")

        # Load pre-trained ResNet18
        # We use the default weights (IMAGENET1K_V1)
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)

        # Remove the fully connected layer to get embeddings (512 dim)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.model.to(self.device)
        self.model.eval()

        # Define standard ImageNet transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_transform(self):
        return self.transform

    def encode_batch(self, images):
        """
        Encodes a batch of images.
        """
        with torch.no_grad():
            images = images.to(self.device)
            # Output shape: (Batch, 512, 1, 1)
            embeddings = self.model(images)
            # Flatten to (Batch, 512)
            embeddings = embeddings.view(embeddings.size(0), -1)
            # L2 Normalize embeddings for Cosine Similarity later
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()


def generate_embeddings(load_cached_data=True, batch_size=256):
    """
    Generates or loads image embeddings for all articles.

    Args:
        load_cached_data (bool): Whether to load from cache if available.
        batch_size (int): Batch size for inference.

    Returns:
        tuple: (embeddings, article_ids)
            embeddings (np.ndarray): Shape (N, 512)
            article_ids (np.ndarray): Shape (N,)
    """
    # 1. Check Cache
    if load_cached_data:
        if (
            config.IMAGE_EMBEDDINGS_PATH.exists()
            and config.ARTICLE_ID_MAP_PATH.exists()
        ):
            print(f"Loading cached embeddings from {config.IMAGE_EMBEDDINGS_PATH}...")
            embeddings = np.load(config.IMAGE_EMBEDDINGS_PATH)
            article_ids = np.load(config.ARTICLE_ID_MAP_PATH)
            return embeddings, article_ids
        else:
            print("Cache not found. Generating embeddings from scratch...")
    else:
        print("Force regeneration. Generating embeddings from scratch...")

    # 2. Load Data
    articles_df = data_loader.load_articles(load_cached_data=True)

    # Filter articles that actually have image files to save time
    # Note: Checking file existence one by one is slow.
    # We will rely on the Dataset to handle missing files gracefully,
    # but we can do a quick pre-filter if we assume standard paths.
    # Given the environment, we'll process all and filter failures during the loop.

    article_ids_all = articles_df["article_id"].values
    image_paths_all = articles_df["image_path"].values

    # 3. Setup Model and Dataset
    encoder = VisualEncoder()
    dataset = HNMImageDataset(
        article_ids=article_ids_all,
        image_paths=image_paths_all,
        transform=encoder.get_transform(),
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 4. Inference Loop
    valid_embeddings = []
    valid_article_ids = []

    print(f"Starting inference on {len(dataset)} items...")

    processed_count = 0

    for images, ids, success_flags in dataloader:
        # Filter out failed loads within the batch
        # success_flags is a tensor of booleans
        mask = success_flags.bool()

        if not mask.any():
            continue

        valid_imgs = images[mask]
        valid_ids_batch = ids[mask]

        # Encode
        emb_batch = encoder.encode_batch(valid_imgs)

        valid_embeddings.append(emb_batch)
        valid_article_ids.append(valid_ids_batch.numpy())

        processed_count += len(valid_ids_batch)
        if processed_count % 10000 == 0:
            print(f"Processed {processed_count} images...")

    # 5. Aggregate Results
    if valid_embeddings:
        all_embeddings = np.concatenate(valid_embeddings, axis=0)
        all_article_ids = np.concatenate(valid_article_ids, axis=0)
    else:
        print("Warning: No valid images found.")
        all_embeddings = np.array([]).reshape(0, 512)
        all_article_ids = np.array([])

    print(f"Finished. Total valid embeddings: {len(all_embeddings)}")

    # 6. Save to Cache
    config.WORKING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving embeddings to {config.IMAGE_EMBEDDINGS_PATH}...")
    np.save(config.IMAGE_EMBEDDINGS_PATH, all_embeddings)

    print(f"Saving article ID map to {config.ARTICLE_ID_MAP_PATH}...")
    np.save(config.ARTICLE_ID_MAP_PATH, all_article_ids)

    return all_embeddings, all_article_ids
