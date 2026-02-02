import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from library import config, utils, dataset


class FeatureExtractor:
    """
    Handles the extraction of visual features from BSON datasets using a frozen ResNet-50.
    Implements caching to avoid redundant computation and prepares hierarchical labels.
    """

    def __init__(self):
        self.device = config.DEVICE
        self.encoder = utils.HierarchyEncoder()
        self.encoder.prepare()  # Ensure encoder is fitted and saved

        # Initialize Model (Lazy loading in extract method to save memory if cached)
        self.model = None

    def _load_model(self):
        """Initializes the ResNet-50 backbone if not already loaded."""
        if self.model is not None:
            return

        print("Initializing ResNet-50 backbone...")
        # Load pre-trained ResNet-50
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        base_model = torchvision.models.resnet50(weights=weights)

        # Replace the classification head with Identity to get the 2048-dim feature vector
        # ResNet forward: conv1 -> ... -> avgpool -> flatten -> fc
        # We want the output of flatten (which is input to fc)
        base_model.fc = nn.Identity()

        self.model = base_model.to(self.device)
        self.model.eval()

    def extract_features(self, split, load_cached_data=True):
        """
        Extracts features for the specified split (train/val/test).

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from disk first.
        """
        # Determine paths based on split
        if split == "train":
            feat_path = config.TRAIN_FEATURES_PATH
            l1_path = config.TRAIN_LABELS_L1_PATH
            l2_path = config.TRAIN_LABELS_L2_PATH
            l3_path = config.TRAIN_LABELS_L3_PATH
            bson_path = config.TRAIN_BSON_PATH
        elif split == "val":
            feat_path = config.VAL_FEATURES_PATH
            l1_path = config.VAL_LABELS_L1_PATH
            l2_path = config.VAL_LABELS_L2_PATH
            l3_path = config.VAL_LABELS_L3_PATH
            bson_path = config.TRAIN_BSON_PATH  # Val is subset of train.bson
        elif split == "test":
            feat_path = config.TEST_FEATURES_PATH
            ids_path = config.TEST_IDS_PATH
            bson_path = config.TEST_BSON_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Check Cache
        if load_cached_data:
            if split == "test":
                if os.path.exists(feat_path) and os.path.exists(ids_path):
                    print(f"[{split}] Loading cached features from {feat_path}...")
                    return
            else:
                if (
                    os.path.exists(feat_path)
                    and os.path.exists(l1_path)
                    and os.path.exists(l2_path)
                    and os.path.exists(l3_path)
                ):
                    print(f"[{split}] Loading cached features from {feat_path}...")
                    return

        print(f"[{split}] Starting feature extraction...")

        # Ensure output directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)

        # Load Metadata
        metadata = utils.load_metadata(split)

        # Initialize Dataset and DataLoader
        # We use a larger batch size for inference as we don't need to store gradients
        # A batch size of 256 products (approx 500-1000 images) fits on A100
        ds = dataset.BSONInferenceDataset(metadata, bson_path)
        loader = DataLoader(
            ds,
            batch_size=256,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=dataset.inference_collate,
            pin_memory=True,
        )

        # Initialize Model
        self._load_model()

        # Pre-allocate feature array to avoid memory fragmentation
        # Shape: (N_samples, 2048)
        num_samples = len(metadata)
        features_array = np.zeros((num_samples, config.INPUT_DIM), dtype=np.float32)

        # For test set, we also need to store IDs explicitly extracted from dataloader
        # to ensure alignment, though metadata order is preserved.
        collected_ids = []

        # Inference Loop
        ptr = 0
        with torch.no_grad():
            for images_list, ids_list in loader:
                # images_list is a list of tensors, each (N_imgs, C, H, W)
                # We flatten this to a single batch for the GPU: (Total_Imgs, C, H, W)

                # Track number of images per product to split later
                lengths = [img.shape[0] for img in images_list]

                # Concatenate all images
                batch_tensor = torch.cat(images_list, dim=0).to(self.device)

                # Mixed Precision Inference for speed
                with torch.amp.autocast("cuda"):
                    batch_feats = self.model(batch_tensor)  # (Total_Imgs, 2048)

                # Split back into products and pool
                cursor = 0
                batch_embeddings = []

                for length in lengths:
                    # Extract features for this product
                    prod_feats = batch_feats[cursor : cursor + length]
                    cursor += length

                    # Mean Pooling: Average feature vectors of all images for this product
                    # Shape: (1, 2048)
                    prod_emb = torch.mean(prod_feats, dim=0)
                    batch_embeddings.append(prod_emb)

                # Stack batch embeddings: (Batch_Size, 2048)
                batch_embeddings = torch.stack(batch_embeddings).cpu().numpy()

                # Store in pre-allocated array
                batch_size = len(ids_list)
                features_array[ptr : ptr + batch_size] = batch_embeddings
                ptr += batch_size

                if split == "test":
                    collected_ids.extend(ids_list)

        print(f"[{split}] Extraction complete. Saving to disk...")

        # Save Features
        np.save(feat_path, features_array)

        # Save Labels / IDs
        if split == "test":
            np.save(ids_path, np.array(collected_ids))
        else:
            # For train/val, generate hierarchical labels
            print(f"[{split}] Generating hierarchical labels...")
            category_ids = metadata["category_id"].values
            l3, l2, l1 = self.encoder.transform(category_ids)

            np.save(l1_path, l1)
            np.save(l2_path, l2)
            np.save(l3_path, l3)

        print(f"[{split}] Data saved successfully.")

    def run_all(self, load_cached_data=True):
        """Runs extraction for all splits."""
        self.extract_features("train", load_cached_data=load_cached_data)
        self.extract_features("val", load_cached_data=load_cached_data)
        self.extract_features("test", load_cached_data=load_cached_data)
