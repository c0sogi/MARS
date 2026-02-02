import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import AutoModel, AutoConfig
from PIL import Image
from library.config import Config
from library.utils import setup_logger


class LeafDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        self.df = pd.read_csv(metadata_path)
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image and convert to RGB (models expect 3 channels)
        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_id


class FeatureExtractor:
    def __init__(self):
        self.device = Config.DEVICE
        self.logger = setup_logger(
            os.path.join(Config.WORKING_DIR, "feature_extraction.log")
        )

        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Base transform: Resize -> ToTensor (0-1 range)
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

    def _get_models(self):
        """Loads DINOv2 and ConvNeXt models."""
        self.logger.info("Loading models...")

        # Load DINOv2
        dino_config = AutoConfig.from_pretrained(Config.MODEL_DINO_ID)
        dino_model = AutoModel.from_pretrained(Config.MODEL_DINO_ID, config=dino_config)
        dino_model.to(self.device)
        dino_model.eval()

        # Load ConvNeXt
        conv_config = AutoConfig.from_pretrained(Config.MODEL_CONVNEXT_ID)
        conv_model = AutoModel.from_pretrained(
            Config.MODEL_CONVNEXT_ID, config=conv_config
        )
        conv_model.to(self.device)
        conv_model.eval()

        return dino_model, conv_model

    def _process_batch(self, images, dino_model, conv_model):
        """
        Generates 12 rotations for the batch and extracts features.
        Input: images (B, C, H, W) tensor on CPU or GPU (0-1 range, unnormalized)
        Output: dino_feats (B, 12, D1), conv_feats (B, 12, D2)
        """
        batch_size = images.size(0)
        views = []

        # Generate 12 views for each image in the batch
        # Config.ROTATION_ANGLES = [0, 30, ..., 330]
        for angle in Config.ROTATION_ANGLES:
            # Rotate with white fill (1.0) because images are black leaves on white
            # TF.rotate expects tensor [C, H, W]
            # We can rotate the whole batch at once if we pass the batch tensor
            # torchvision functional rotate supports batched tensors
            rotated_imgs = TF.rotate(images, angle=angle, fill=[1.0])

            # Normalize after rotation to avoid padding artifacts with normalized values
            normalized_imgs = self.normalize(rotated_imgs)
            views.append(normalized_imgs)

        # Stack views: (12, B, C, H, W) -> (B*12, C, H, W)
        # We interleave to keep views of same image together or just stack?
        # Let's stack such that shape is (Batch * 12, C, H, W)
        # To make reshaping easier:
        #   First dim 0 is angle 0 for all images
        #   Second dim 1 is angle 30 for all images...
        #   This is (12, B, C, H, W).
        #   Permute to (B, 12, C, H, W) then flatten to (B*12, C, H, W)

        multi_view_batch = torch.stack(views, dim=1)  # (B, 12, C, H, W)
        multi_view_batch = multi_view_batch.view(
            batch_size * Config.NUM_VIEWS, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
        )
        multi_view_batch = multi_view_batch.to(self.device)

        with torch.no_grad():
            # DINOv2 Inference
            dino_out = dino_model(pixel_values=multi_view_batch)
            # Use CLS token (index 0) from last_hidden_state
            # Shape: (B*12, Seq, Dim) -> (B*12, Dim)
            if hasattr(dino_out, "last_hidden_state"):
                dino_emb = dino_out.last_hidden_state[:, 0, :]
            else:
                dino_emb = dino_out.pooler_output

            # ConvNeXt Inference
            conv_out = conv_model(pixel_values=multi_view_batch)
            # Use pooler_output (Global Avg Pool)
            # Shape: (B*12, Dim)
            if (
                hasattr(conv_out, "pooler_output")
                and conv_out.pooler_output is not None
            ):
                conv_emb = conv_out.pooler_output
            else:
                # Fallback if pooler_output is None (depends on config)
                # ConvNeXt last_hidden_state is (B, C, H, W). Mean over H,W.
                conv_emb = conv_out.last_hidden_state.mean(dim=[-2, -1])

        # Reshape back to (B, 12, Dim)
        dino_feats = dino_emb.view(batch_size, Config.NUM_VIEWS, -1).cpu().numpy()
        conv_feats = conv_emb.view(batch_size, Config.NUM_VIEWS, -1).cpu().numpy()

        return dino_feats, conv_feats

    def extract_features(self, metadata_path, subset_name, load_cached_data=True):
        """
        Extracts features for the given dataset subset.

        Args:
            metadata_path (str): Path to metadata CSV.
            subset_name (str): Name of subset (e.g., 'train', 'val', 'test') for cache naming.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            ids (np.array): Image IDs.
            dino_features (np.array): Shape (N, 12, 1024)
            conv_features (np.array): Shape (N, 12, 1536)
        """
        cache_dino = os.path.join(
            Config.WORKING_DIR, f"{subset_name}_dino_features.npy"
        )
        cache_conv = os.path.join(
            Config.WORKING_DIR, f"{subset_name}_convnext_features.npy"
        )
        cache_ids = os.path.join(Config.WORKING_DIR, f"{subset_name}_ids.npy")

        # Check cache
        if load_cached_data:
            if (
                os.path.exists(cache_dino)
                and os.path.exists(cache_conv)
                and os.path.exists(cache_ids)
            ):
                self.logger.info(f"Loading cached features for {subset_name}...")
                return (np.load(cache_ids), np.load(cache_dino), np.load(cache_conv))

        self.logger.info(f"Starting feature extraction for {subset_name}...")

        # Initialize Dataset and Loader
        dataset = LeafDataset(metadata_path, transform=self.base_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE_EXTRACTION,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load Models
        dino_model, conv_model = self._get_models()

        all_ids = []
        all_dino_feats = []
        all_conv_feats = []

        for batch_imgs, batch_ids in dataloader:
            # batch_imgs is (B, 3, H, W)
            dino_f, conv_f = self._process_batch(batch_imgs, dino_model, conv_model)

            all_ids.append(batch_ids.numpy())
            all_dino_feats.append(dino_f)
            all_conv_feats.append(conv_f)

        # Concatenate
        final_ids = np.concatenate(all_ids, axis=0)
        final_dino = np.concatenate(all_dino_feats, axis=0)
        final_conv = np.concatenate(all_conv_feats, axis=0)

        self.logger.info(
            f"Extraction complete. Shapes: DINO {final_dino.shape}, ConvNeXt {final_conv.shape}"
        )

        # Save to cache
        np.save(cache_ids, final_ids)
        np.save(cache_dino, final_dino)
        np.save(cache_conv, final_conv)
        self.logger.info(f"Features cached to {Config.WORKING_DIR}")

        # Cleanup to free GPU memory
        del dino_model
        del conv_model
        torch.cuda.empty_cache()

        return final_ids, final_dino, final_conv
