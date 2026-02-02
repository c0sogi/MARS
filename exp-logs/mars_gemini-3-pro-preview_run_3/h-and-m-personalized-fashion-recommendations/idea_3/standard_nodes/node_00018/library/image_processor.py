import os
import cv2
import torch
import timm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict, Tuple, Optional

from library import config
from library import utils


class ArticleImageDataset(Dataset):
    """
    Dataset class for loading article images.
    """

    def __init__(
        self, article_ids: np.ndarray, image_dir: str, img_size: Tuple[int, int]
    ):
        self.article_ids = article_ids
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Pre-calculate paths to avoid string ops in getitem
        self.image_paths = [self._get_path(aid) for aid in article_ids]

    def _get_path(self, article_id: int) -> str:
        s = str(article_id).zfill(10)
        folder = s[:3]
        return os.path.join(self.image_dir, folder, f"{s}.jpg")

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        aid = self.article_ids[idx]

        try:
            # Open image using PIL
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, aid, True
        except (OSError, FileNotFoundError):
            # Return a zero tensor if image is corrupt or missing (though we filter beforehand)
            return (
                torch.zeros((3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])),
                aid,
                False,
            )


class ImageEmbedder:
    """
    Handles extraction of visual features from article images using a pre-trained CNN.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.IMAGE_BATCH_SIZE
        self.img_size = config.IMAGE_SIZE
        self.model_name = config.IMAGE_MODEL_NAME

        # Cache paths
        self.cache_dir = config.WORKING_DIR
        self.emb_cache_path = self.cache_dir / "article_embeddings.npy"
        self.ids_cache_path = self.cache_dir / "article_ids.npy"

    def _get_model(self):
        print(f"Loading model: {self.model_name}")
        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        model = model.to(self.device)
        model.eval()
        return model

    def _filter_existing_images(self, articles_df: pd.DataFrame) -> np.ndarray:
        """
        Returns an array of article_ids that actually have image files on disk.
        """
        valid_ids = []
        # We assume the directory structure follows the metadata logic
        # Vectorized check is hard with file system, so we do a quick scan or check
        # Given the dataset size, we check existence.

        # Optimization: The metadata generation script already verified paths.
        # However, we must ensure we only try to load existing files.
        # Let's construct expected paths and filter.

        print("Filtering for existing images...")
        # Use simple string manipulation
        aids = articles_df["article_id"].astype(str).str.zfill(10)
        subfolders = aids.str[:3]
        filenames = aids + ".jpg"
        paths = [
            config.INPUT_DIR / "images" / f / n for f, n in zip(subfolders, filenames)
        ]

        # Check existence (this might take a minute for 100k files, but it's safe)
        # To speed up, we can assume most exist, or rely on the dataset try/except block.
        # But for batch processing, it's better to supply only valid paths to DataLoader.

        # Actually, checking 100k file existence can be slow.
        # Strategy: Pass all to Dataset, let Dataset return a flag 'valid'.
        # We filter the output embeddings based on this flag.
        return articles_df["article_id"].values

    def _extract_raw_embeddings(
        self, valid_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference to get embeddings.
        Returns: (embeddings, valid_article_ids)
        """
        dataset = ArticleImageDataset(
            valid_ids, str(config.INPUT_DIR / "images"), self.img_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.N_CPUS,
            pin_memory=True,
        )

        model = self._get_model()

        embeddings_list = []
        ids_list = []

        print(f"Starting inference on {len(valid_ids)} potential images...")

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for imgs, aids, valids in tqdm(dataloader, desc="Extracting Embeddings"):
                imgs = imgs.to(self.device)

                # Forward pass
                features = model(imgs)

                # Move to CPU
                features = features.cpu().numpy()
                aids = aids.numpy()
                valids = valids.numpy()

                # Filter out invalid loads
                valid_mask = valids.astype(bool)
                if valid_mask.any():
                    embeddings_list.append(features[valid_mask])
                    ids_list.append(aids[valid_mask])

        if not embeddings_list:
            return np.array([]), np.array([])

        return np.vstack(embeddings_list), np.concatenate(ids_list)

    def get_embeddings(
        self, articles_df: pd.DataFrame, load_cached_data: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Main method to retrieve embeddings.
        1. Checks cache.
        2. If missing, computes embeddings for existing images.
        3. Imputes missing embeddings using Product Group Average.
        4. Returns dictionary {article_id: embedding_vector}.
        """
        utils.seed_everything(config.SEED)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load or Compute Raw Embeddings
        if (
            load_cached_data
            and self.emb_cache_path.exists()
            and self.ids_cache_path.exists()
        ):
            print(f"Loading embeddings from cache: {self.emb_cache_path}")
            raw_embeddings = np.load(self.emb_cache_path)
            raw_ids = np.load(self.ids_cache_path)
        else:
            print("Cache not found or ignored. Computing embeddings...")
            all_ids = articles_df["article_id"].values
            raw_embeddings, raw_ids = self._extract_raw_embeddings(all_ids)

            # Save to cache
            print(f"Saving embeddings to cache: {self.cache_dir}")
            np.save(self.emb_cache_path, raw_embeddings)
            np.save(self.ids_cache_path, raw_ids)

        # 2. Create Mapping for Existing Images
        # Map article_id -> embedding
        print("Constructing embedding dictionary...")
        emb_dim = (
            raw_embeddings.shape[1]
            if len(raw_embeddings) > 0
            else config.IMAGE_EMBEDDING_DIM
        )
        id_to_emb = dict(zip(raw_ids, raw_embeddings))

        # 3. Imputation Strategy
        print("Handling missing images (Imputation by Product Group)...")

        # Join embeddings with product group info
        # Create a temporary DF for calculation
        emb_df = pd.DataFrame(raw_embeddings)
        emb_df["article_id"] = raw_ids

        # Merge with articles metadata to get product_group_name
        # We only need article_id and product_group_name
        meta_df = articles_df[["article_id", "product_group_name"]].copy()
        merged_df = meta_df.merge(emb_df, on="article_id", how="left")

        # Calculate mean embedding per product group
        # Drop article_id and group by product_group_name
        # Columns 0...511 are the embedding features
        feature_cols = [c for c in merged_df.columns if isinstance(c, int)]

        # If no embeddings were found at all (edge case), create zero vector
        if not feature_cols:
            print("Warning: No images processed. Returning zero vectors.")
            zero_vec = np.zeros(config.IMAGE_EMBEDDING_DIM, dtype=np.float32)
            return {aid: zero_vec for aid in articles_df["article_id"].values}

        group_means = merged_df.groupby("product_group_name")[feature_cols].mean()

        # Global mean for fallback (if a group has 0 images)
        global_mean = merged_df[feature_cols].mean().values
        if np.isnan(global_mean).any():
            global_mean = np.zeros(emb_dim, dtype=np.float32)

        # 4. Fill Missing Values
        final_embeddings = {}

        # Iterate through all requested articles
        # Optimization: Pre-convert group means to dict
        group_mean_dict = {k: v.values for k, v in group_means.iterrows()}

        # Convert meta_df to records for fast iteration
        records = meta_df.to_dict("records")

        for row in tqdm(records, desc="Finalizing Embeddings"):
            aid = row["article_id"]
            group = row["product_group_name"]

            if aid in id_to_emb:
                final_embeddings[aid] = id_to_emb[aid]
            else:
                # Impute
                if (
                    group in group_mean_dict
                    and not np.isnan(group_mean_dict[group]).any()
                ):
                    final_embeddings[aid] = group_mean_dict[group]
                else:
                    final_embeddings[aid] = global_mean

        return final_embeddings
