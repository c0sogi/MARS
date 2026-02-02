import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from library.config import Config
from library.utils import Timer, seed_everything, print_memory_usage


class ArticleImageDataset(Dataset):
    """
    PyTorch Dataset to load article images based on mapped indices.
    """

    def __init__(self, article_map, img_dir, transform=None):
        self.article_map = article_map
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.article_map)

    def __getitem__(self, idx):
        # Get original article ID from the map
        original_id = self.article_map[idx]

        # Construct path: images/012/0123456789.jpg
        s = str(original_id).zfill(10)
        folder = s[:3]
        img_path = self.img_dir / folder / f"{s}.jpg"

        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    img = Image.open(f).convert("RGB")

                if self.transform:
                    img = self.transform(img)
                return img, 1  # 1 indicates valid image
            except Exception:
                # Fallback for corrupted images
                pass

        # Return zero tensor if missing or corrupt
        # Shape must match transform output: 3, 224, 224
        return (
            torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=torch.float32),
            0,
        )  # 0 indicates missing


class VisualEncoder:
    """
    Generates visual embeddings for articles using a pre-trained ResNet.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.batch_size = Config.IMAGE_BATCH_SIZE

        # Define Transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self):
        """Loads ResNet18 and removes the classification head."""
        print("Loading ResNet18 model...")
        # Use weights=None and load state_dict if needed, or use pretrained=True (deprecated)
        # or weights='IMAGENET1K_V1'. The environment likely supports weights.
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except:
            # Fallback for older torchvision versions
            model = models.resnet18(pretrained=True)

        # Remove the fully connected layer to get embeddings
        # ResNet structure: ... -> avgpool -> fc
        # We want the output of avgpool (flattened)
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)

        model.to(self.device)
        model.eval()
        return model

    def generate_embeddings(self, load_cached_data: bool = True):
        """
        Main method to generate or load embeddings.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            np.ndarray: Matrix of shape (num_articles, 512)
        """
        # 1. Check Cache
        if load_cached_data and Config.CACHE_IMAGE_EMBEDDINGS.exists():
            print("Loading image embeddings from cache...")
            with Timer("Load Embeddings Cache"):
                embeddings = np.load(Config.CACHE_IMAGE_EMBEDDINGS)
            print_memory_usage("After Embeddings Load")
            return embeddings

        print("Generating image embeddings from scratch...")

        # 2. Load Article Map
        if not Config.CACHE_ARTICLE_MAP.exists():
            raise FileNotFoundError(
                f"Article map not found at {Config.CACHE_ARTICLE_MAP}. Run DataLoader first."
            )

        article_map = np.load(Config.CACHE_ARTICLE_MAP, allow_pickle=True)
        num_articles = len(article_map)
        print(f"Total articles to process: {num_articles}")

        # 3. Setup Dataset and Loader
        dataset = ArticleImageDataset(
            article_map=article_map, img_dir=Config.IMAGES_DIR, transform=self.transform
        )

        # Adjust workers based on CPU availability
        num_workers = min(Config.NUM_WORKERS, 4)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Must be False to maintain index alignment
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        # 4. Inference Loop
        model = self._load_model()

        # Pre-allocate array
        # ResNet18 avgpool output is 512
        embedding_dim = 512
        embeddings = np.zeros((num_articles, embedding_dim), dtype=np.float32)

        start_idx = 0

        # Debug limit
        limit = num_articles
        if Config.DEBUG:
            limit = min(num_articles, Config.DEBUG_SAMPLE_SIZE)
            print(f"Debug Mode: Processing only first {limit} images.")

        with torch.no_grad(), Timer("Image Inference"):
            for batch_imgs, valid_mask in tqdm(loader, desc="Encoding Images"):
                if start_idx >= limit:
                    break

                batch_imgs = batch_imgs.to(self.device)

                # Forward pass
                # Output shape: (Batch, 512, 1, 1)
                features = model(batch_imgs)
                features = features.squeeze(-1).squeeze(-1)  # (Batch, 512)

                # Normalize (L2)
                features = torch.nn.functional.normalize(features, p=2, dim=1)

                # Handle missing images: Zero out embeddings where valid_mask is 0
                # valid_mask is on CPU, move to device or apply after moving to CPU
                # Easier to apply on CPU numpy array

                batch_emb = features.cpu().numpy()
                batch_valid = valid_mask.numpy()

                # Zero out invalid
                batch_emb[batch_valid == 0] = 0.0

                # Store
                batch_size_curr = batch_emb.shape[0]
                end_idx = start_idx + batch_size_curr

                # Handle debug truncation within batch
                if Config.DEBUG and end_idx > limit:
                    keep = limit - start_idx
                    embeddings[start_idx:limit] = batch_emb[:keep]
                    break
                else:
                    embeddings[start_idx:end_idx] = batch_emb

                start_idx = end_idx

        # 5. Save to Cache
        print("Saving image embeddings to cache...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(Config.CACHE_IMAGE_EMBEDDINGS, embeddings)

        # Clean up
        del model, loader, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_memory_usage("After Embedding Generation")
        return embeddings
