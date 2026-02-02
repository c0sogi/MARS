import os
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


class LeafMultiViewDataset(Dataset):
    """
    Dataset class that loads an image and generates multiple rotated views.
    """

    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.num_rotations = Config.NUM_ROTATIONS
        # Angles: 0, 30, 60, ..., 330
        self.angles = np.linspace(0, 360, self.num_rotations, endpoint=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]

        # Construct full path. Metadata contains relative path 'images/{id}.jpg'
        # Config.INPUT_DIR is './input'.
        # We need to be careful: row['file_path'] is like 'images/10.jpg'
        # Config.INPUT_DIR is './input'
        # So full path is ./input/images/10.jpg
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # Open image and convert to RGB (standard for pre-trained models)
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for missing images (though verification script passed)
            print(f"Warning: Could not load image {img_path}. Using black image.")
            img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (255, 255, 255))

        views = []
        for angle in self.angles:
            # Rotate image.
            # PIL rotate is counter-clockwise.
            # fillcolor=(255, 255, 255) ensures white background for binary leaves
            img_rotated = img.rotate(
                angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)
            )

            if self.transform:
                views.append(self.transform(img_rotated))
            else:
                views.append(transforms.ToTensor()(img_rotated))

        # Stack views: (12, 3, H, W)
        views_tensor = torch.stack(views)

        return {"id": img_id, "views": views_tensor}


class FeatureExtractor:
    """
    Handles deep feature extraction using DINOv2 and ConvNeXt.
    Implements caching to avoid redundant computation.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.img_size = Config.IMG_SIZE

        # Define Transforms
        # Standard ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size, self.img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _get_model(self, model_name):
        """Helper to load a model from timm."""
        print(f"Loading model: {model_name}...")
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model = model.to(self.device)
        model.eval()
        return model

    def extract_features(self, df, dataset_name, load_cached_data=True):
        """
        Extracts features for the given dataframe.

        Args:
            df (pd.DataFrame): Metadata dataframe containing 'id' and 'file_path'.
            dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing 'ids', 'dino_features', 'conv_features'.
        """
        Config.make_dirs()

        # Define cache paths
        cache_ids_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_ids.npy")
        cache_dino_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_dino.npy")
        cache_conv_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_conv.npy")

        # 1. Try to load from cache
        if load_cached_data:
            if (
                os.path.exists(cache_ids_path)
                and os.path.exists(cache_dino_path)
                and os.path.exists(cache_conv_path)
            ):
                print(
                    f"Loading cached features for {dataset_name} from {Config.WORKING_DIR}..."
                )
                ids = np.load(cache_ids_path)
                dino_feats = np.load(cache_dino_path)
                conv_feats = np.load(cache_conv_path)
                return {
                    "ids": ids,
                    "dino_features": dino_feats,
                    "conv_features": conv_feats,
                }
            else:
                print(f"Cache missing for {dataset_name}. Starting extraction...")
        else:
            print(
                f"Force extraction enabled for {dataset_name}. Starting extraction..."
            )

        # 2. Setup Models
        dino_model = self._get_model(Config.MODEL_DINO)
        conv_model = self._get_model(Config.MODEL_CONVNEXT)

        # 3. Setup DataLoader
        dataset = LeafMultiViewDataset(df, Config.INPUT_DIR, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_ids = []
        all_dino_feats = []
        all_conv_feats = []

        # 4. Inference Loop
        with torch.no_grad():
            for batch in dataloader:
                ids = batch["id"]
                # views shape: (B, 12, 3, H, W)
                views = batch["views"].to(self.device)

                B, V, C, H, W = views.shape

                # Flatten views into batch dimension for efficient inference
                # (B*12, 3, H, W)
                flat_views = views.view(B * V, C, H, W)

                # --- DINOv2 Inference ---
                # Output shape: (B*12, Embed_Dim)
                dino_out = dino_model(flat_views)
                # Reshape back: (B, 12, Embed_Dim)
                dino_out = dino_out.view(B, V, -1).cpu().numpy()

                # --- ConvNeXt Inference ---
                conv_out = conv_model(flat_views)
                conv_out = conv_out.view(B, V, -1).cpu().numpy()

                all_ids.extend(ids.numpy())
                all_dino_feats.append(dino_out)
                all_conv_feats.append(conv_out)

        # 5. Concatenate and Save
        all_ids = np.array(all_ids)
        all_dino_feats = np.concatenate(all_dino_feats, axis=0)
        all_conv_feats = np.concatenate(all_conv_feats, axis=0)

        print(f"Saving features for {dataset_name} to {Config.WORKING_DIR}...")
        np.save(cache_ids_path, all_ids)
        np.save(cache_dino_path, all_dino_feats)
        np.save(cache_conv_path, all_conv_feats)

        # Clean up models to free GPU memory
        del dino_model
        del conv_model
        torch.cuda.empty_cache()

        return {
            "ids": all_ids,
            "dino_features": all_dino_feats,
            "conv_features": all_conv_feats,
        }
