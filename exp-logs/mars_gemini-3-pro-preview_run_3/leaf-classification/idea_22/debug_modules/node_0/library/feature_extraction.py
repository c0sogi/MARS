import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from library.config import Config
from library.utils import seed_everything, save_array, load_array


class LeafMultiViewDataset(Dataset):
    """
    Dataset class that loads an image and generates 12 equidistant rotated views.
    """

    def __init__(self, metadata_df, input_dir, transform=None):
        self.df = metadata_df
        self.input_dir = input_dir
        self.transform = transform
        # Generate angles: 0, 30, 60, ..., 330 for NUM_ROTATIONS=12
        step = 360 / Config.NUM_ROTATIONS
        self.angles = [i * step for i in range(Config.NUM_ROTATIONS)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # file_path is relative to input_dir (e.g., "images/123.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image and convert to RGB (models expect 3 channels)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for missing images (should not happen based on metadata checks)
            print(f"Warning: Could not load {img_path}, using blank image.")
            img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (255, 255, 255))

        views = []
        for angle in self.angles:
            # Rotate image using bicubic interpolation.
            # fillcolor=(255, 255, 255) ensures the background remains white when corners are exposed.
            # expand=False keeps the original image dimensions.
            img_rot = img.rotate(
                angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)
            )

            if self.transform:
                views.append(self.transform(img_rot))
            else:
                views.append(transforms.ToTensor()(img_rot))

        # Stack views: (Num_Rotations, 3, H, W)
        return torch.stack(views), row["id"]


class DualStreamExtractor:
    """
    Extracts features using DINOv2 (Global Geometry) and ConvNeXt (Local Texture) models.
    """

    def __init__(self, device=None):
        seed_everything()
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"DualStreamExtractor initialized on {self.device}")

    def extract_features(self, load_cached_data=True):
        """
        Extracts features for Train, Validation, and Test sets.
        Returns a dictionary containing features and IDs for all splits.
        """
        # Define cache file paths
        cache_files = {
            "train_dino": "train_dino_features.npy",
            "train_conv": "train_conv_features.npy",
            "train_ids": "train_ids.npy",
            "val_dino": "val_dino_features.npy",
            "val_conv": "val_conv_features.npy",
            "val_ids": "val_ids.npy",
            "test_dino": "test_dino_features.npy",
            "test_conv": "test_conv_features.npy",
            "test_ids": "test_ids.npy",
        }

        # Check if we should and can load from cache
        if load_cached_data:
            cached_data = {}
            all_found = True
            for key, filename in cache_files.items():
                path = os.path.join(Config.WORKING_DIR, filename)
                arr = load_array(path)
                if arr is None:
                    all_found = False
                    break
                cached_data[key] = arr

            if all_found:
                print("Loaded all features from cache.")
                return cached_data

        print("Feature extraction started (Cache miss or force reload)...")

        # Load Metadata
        train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
        test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

        # Debugging limit
        if Config.DEBUG_SAMPLE_SIZE:
            print(
                f"Debug Mode: Limiting to {Config.DEBUG_SAMPLE_SIZE} samples per split."
            )
            train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
            val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
            test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Define Transforms
        # ImageNet normalization is standard for pretrained timm models
        transform = transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Prepare DataLoaders
        # Using pin_memory=True for faster host-to-device transfer
        train_loader = DataLoader(
            LeafMultiViewDataset(train_df, Config.INPUT_DIR, transform),
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            LeafMultiViewDataset(val_df, Config.INPUT_DIR, transform),
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            LeafMultiViewDataset(test_df, Config.INPUT_DIR, transform),
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load Models
        print(f"Loading DINOv2 model: {Config.DINO_MODEL}")
        dino_model = timm.create_model(
            Config.DINO_MODEL, pretrained=True, num_classes=0
        ).to(self.device)
        dino_model.eval()

        print(f"Loading ConvNeXt model: {Config.CONVNEXT_MODEL}")
        conv_model = timm.create_model(
            Config.CONVNEXT_MODEL, pretrained=True, num_classes=0
        ).to(self.device)
        conv_model.eval()

        # Run Inference
        print("Extracting features for Training set...")
        train_dino, train_conv, train_ids = self._process_loader(
            train_loader, dino_model, conv_model
        )

        print("Extracting features for Validation set...")
        val_dino, val_conv, val_ids = self._process_loader(
            val_loader, dino_model, conv_model
        )

        print("Extracting features for Test set...")
        test_dino, test_conv, test_ids = self._process_loader(
            test_loader, dino_model, conv_model
        )

        # Construct result dictionary
        results = {
            "train_dino": train_dino,
            "train_conv": train_conv,
            "train_ids": train_ids,
            "val_dino": val_dino,
            "val_conv": val_conv,
            "val_ids": val_ids,
            "test_dino": test_dino,
            "test_conv": test_conv,
            "test_ids": test_ids,
        }

        # Save to cache
        print("Saving features to cache...")
        for key, filename in cache_files.items():
            save_array(results[key], os.path.join(Config.WORKING_DIR, filename))

        return results

    def _process_loader(self, loader, dino_model, conv_model):
        """
        Internal method to run inference on a DataLoader.
        """
        dino_feats = []
        conv_feats = []
        ids_list = []

        with torch.no_grad():
            for images, ids in loader:
                # images shape: (Batch, Num_Rotations, 3, H, W)
                B, R, C, H, W = images.shape

                # Flatten batch and rotations for efficient processing
                # (Batch * Num_Rotations, 3, H, W)
                flat_images = images.view(B * R, C, H, W).to(self.device)

                # Extract DINO features
                # Output: (B*R, Embed_Dim)
                d_out = dino_model(flat_images)

                # Extract ConvNeXt features
                # Output: (B*R, Embed_Dim)
                c_out = conv_model(flat_images)

                # Reshape back to (Batch, Num_Rotations, Embed_Dim)
                # Move to CPU to save GPU memory
                d_out = d_out.view(B, R, -1).cpu().numpy()
                c_out = c_out.view(B, R, -1).cpu().numpy()

                dino_feats.append(d_out)
                conv_feats.append(c_out)
                ids_list.append(ids.numpy())

        return (
            np.concatenate(dino_feats),
            np.concatenate(conv_feats),
            np.concatenate(ids_list),
        )
