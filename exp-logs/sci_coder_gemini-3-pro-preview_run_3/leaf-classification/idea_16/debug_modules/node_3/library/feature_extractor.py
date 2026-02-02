import os
import numpy as np
import pandas as pd
import torch
import timm
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything


class LeafDataset(Dataset):
    """
    Dataset class for loading leaf images based on metadata CSVs.
    """

    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        # Metadata file_path is relative (e.g., "images/123.jpg")
        rel_path = row["file_path"]
        full_path = os.path.join(self.img_dir, rel_path)

        # Open image and ensure RGB (3 channels)
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a blank white image in case of error to prevent crash
            img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (255, 255, 255))

        return img, img_id


class ImageEmbedder:
    """
    Handles loading of DINOv2 and ConvNeXt models and extraction of features
    from multi-view rotated images.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dino_model = None
        self.convnext_model = None

        # Preprocessing transforms
        self.transform_resize = transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE))
        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _load_models(self):
        """
        Loads the pretrained models onto the device if not already loaded.
        """
        if self.dino_model is None:
            print(f"Loading DINOv2 model: {Config.DINO_MODEL_NAME}...")
            self.dino_model = timm.create_model(
                Config.DINO_MODEL_NAME,
                pretrained=True,
                num_classes=0,
                img_size=Config.IMG_SIZE,
            )
            self.dino_model.to(self.device)
            self.dino_model.eval()

        if self.convnext_model is None:
            print(f"Loading ConvNeXt model: {Config.CONVNEXT_MODEL_NAME}...")
            self.convnext_model = timm.create_model(
                Config.CONVNEXT_MODEL_NAME, pretrained=True, num_classes=0
            )
            self.convnext_model.to(self.device)
            self.convnext_model.eval()

    def _process_batch_images(self, images):
        """
        Applies rotation and normalization to a batch of PIL images.

        Args:
            images: List of PIL Images.

        Returns:
            full_batch: Tensor of shape (B * 36, 3, H, W)
            b: Batch size
            v: Number of views (36)
        """
        batch_tensors = []

        for img in images:
            # Resize
            img_resized = self.transform_resize(img)
            # Convert to Tensor (scales to [0, 1])
            img_tensor = transforms.functional.to_tensor(img_resized)

            views = []
            for angle in Config.ROTATION_ANGLES:
                # Rotate image.
                # fill=[1.0] ensures the background remains white (1.0 in float tensor)
                rotated = TF.rotate(img_tensor, angle, fill=[1.0])
                # Normalize
                normalized = self.normalize(rotated)
                views.append(normalized)

            # Stack views for this image: (36, 3, H, W)
            img_views = torch.stack(views)
            batch_tensors.append(img_views)

        # Stack all images in batch: (B, 36, 3, H, W)
        full_batch = torch.stack(batch_tensors)
        b, v, c, h, w = full_batch.shape

        # Flatten batch and views for efficient inference: (B*36, 3, H, W)
        return full_batch.view(b * v, c, h, w), b, v

    def extract_features(
        self, dataset_name, csv_path, load_cached_data=True, batch_size=2
    ):
        """
        Extracts features for the specified dataset.

        Args:
            dataset_name (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
            csv_path (str): Path to the metadata CSV file.
            load_cached_data (bool): If True, attempts to load features from disk.
            batch_size (int): Batch size for inference.

        Returns:
            dict: Dictionary containing 'dino', 'convnext' feature arrays and 'ids'.
        """
        seed_everything()

        # Define cache file paths
        cache_dino = os.path.join(Config.CACHE_DIR, f"{dataset_name}_dino.npy")
        cache_convnext = os.path.join(Config.CACHE_DIR, f"{dataset_name}_convnext.npy")
        cache_ids = os.path.join(Config.CACHE_DIR, f"{dataset_name}_ids.npy")

        # 1. Attempt to load from cache
        if load_cached_data:
            if (
                os.path.exists(cache_dino)
                and os.path.exists(cache_convnext)
                and os.path.exists(cache_ids)
            ):
                print(
                    f"[{dataset_name}] Loading cached features from {Config.CACHE_DIR}..."
                )
                return {
                    "dino": np.load(cache_dino),
                    "convnext": np.load(cache_convnext),
                    "ids": np.load(cache_ids),
                }
            else:
                print(
                    f"[{dataset_name}] Cache not found. Starting feature extraction..."
                )
        else:
            print(f"[{dataset_name}] Ignoring cache. Starting feature extraction...")

        # 2. Setup Models and Data
        self._load_models()

        dataset = LeafDataset(csv_path, Config.INPUT_DIR)

        # Custom collate to handle PIL images
        def collate_fn(batch):
            imgs, ids = zip(*batch)
            return list(imgs), np.array(ids)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        all_dino_feats = []
        all_convnext_feats = []
        all_ids = []

        print(f"Processing {len(dataset)} images from {csv_path}...")

        # 3. Inference Loop
        with torch.no_grad():
            for imgs, ids in tqdm(dataloader, desc=f"Extracting {dataset_name}"):
                # Prepare input: (B*36, 3, H, W)
                input_tensor, b, v = self._process_batch_images(imgs)
                input_tensor = input_tensor.to(self.device)

                # --- DINOv2 Inference ---
                dino_out = self.dino_model(input_tensor)
                # Reshape back to (B, 36, Dim)
                dino_dim = dino_out.shape[1]
                dino_out = dino_out.view(b, v, dino_dim).cpu().numpy()
                all_dino_feats.append(dino_out)

                # --- ConvNeXt Inference ---
                convnext_out = self.convnext_model(input_tensor)
                # Reshape back to (B, 36, Dim)
                convnext_dim = convnext_out.shape[1]
                convnext_out = convnext_out.view(b, v, convnext_dim).cpu().numpy()
                all_convnext_feats.append(convnext_out)

                all_ids.append(ids)

        # 4. Aggregate Results
        final_dino = np.concatenate(all_dino_feats, axis=0)
        final_convnext = np.concatenate(all_convnext_feats, axis=0)
        final_ids = np.concatenate(all_ids, axis=0)

        # 5. Save to Cache
        print(f"[{dataset_name}] Saving features to {Config.CACHE_DIR}...")
        np.save(cache_dino, final_dino)
        np.save(cache_convnext, final_convnext)
        np.save(cache_ids, final_ids)

        return {"dino": final_dino, "convnext": final_convnext, "ids": final_ids}
