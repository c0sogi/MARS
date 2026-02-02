import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import DogDataset, get_transforms
from library.model_factory import get_backbone


class FeatureExtractor:
    """
    Handles the extraction of features (embeddings) from images using
    specified model streams and geometric views. Implements Test Time Augmentation (TTA)
    and caching mechanisms.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self._set_seed()

    def _set_seed(self):
        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)

    def _get_paths(self, stream_name, split_name):
        """
        Resolves the file paths for embeddings, labels, and IDs based on stream and split.
        """
        emb_path = None
        lbl_path = None
        id_path = None

        # Determine Embedding Path
        if stream_name == "stream_a":
            if split_name == "train":
                emb_path = Config.CACHE_A_TRAIN_EMB
            elif split_name == "val":
                emb_path = Config.CACHE_A_VAL_EMB
            elif split_name == "test":
                emb_path = Config.CACHE_A_TEST_EMB
        elif stream_name == "stream_b":
            if split_name == "train":
                emb_path = Config.CACHE_B_TRAIN_EMB
            elif split_name == "val":
                emb_path = Config.CACHE_B_VAL_EMB
            elif split_name == "test":
                emb_path = Config.CACHE_B_TEST_EMB

        # Determine Label/ID Path (Shared)
        if split_name == "train":
            lbl_path = Config.CACHE_TRAIN_LABELS
            id_path = Config.CACHE_TRAIN_IDS
        elif split_name == "val":
            lbl_path = Config.CACHE_VAL_LABELS
            id_path = Config.CACHE_VAL_IDS
        elif split_name == "test":
            # Test set has no valid labels, but we track IDs
            lbl_path = None
            id_path = Config.CACHE_TEST_IDS

        return emb_path, lbl_path, id_path

    def _get_metadata_path(self, split_name):
        if split_name == "train":
            return Config.TRAIN_METADATA_PATH
        if split_name == "val":
            return Config.VAL_METADATA_PATH
        if split_name == "test":
            return Config.TEST_METADATA_PATH
        raise ValueError(f"Unknown split: {split_name}")

    def _run_inference(self, model, loader, view_type):
        """
        Runs inference on a dataloader with TTA.
        Returns embeddings, labels, and ids.
        """
        embeddings_list = []
        labels_list = []
        ids_list = []

        model.eval()

        with torch.no_grad():
            for images, labels, img_ids in loader:
                images = images.to(self.device)

                # --- Test Time Augmentation (TTA) ---

                if view_type in ["global", "standard"]:
                    # Input shape: (B, C, H, W)
                    # 1. Original
                    feat_orig = model(images)

                    # 2. Horizontal Flip
                    images_flip = torch.flip(images, dims=[3])  # Flip width dimension
                    feat_flip = model(images_flip)

                    # Average
                    batch_emb = (feat_orig + feat_flip) / 2.0

                elif view_type == "robust":
                    # Input shape: (B, 5, C, H, W)
                    b, n_crops, c, h, w = images.shape

                    # Flatten to (B*5, C, H, W) for batch processing
                    images_flat = images.view(-1, c, h, w)

                    # 1. Original Crops
                    feat_orig = model(images_flat)  # (B*5, D)

                    # 2. Flipped Crops
                    images_flip = torch.flip(images_flat, dims=[3])
                    feat_flip = model(images_flip)  # (B*5, D)

                    # Average Flip and Original
                    feat_avg_crops = (feat_orig + feat_flip) / 2.0

                    # Reshape back to (B, 5, D) and average over crops
                    # Assuming feature dim is last
                    dim_feat = feat_avg_crops.shape[-1]
                    feat_reshaped = feat_avg_crops.view(b, n_crops, dim_feat)

                    # Average over the 5 crops
                    batch_emb = feat_reshaped.mean(dim=1)  # (B, D)

                else:
                    raise ValueError(f"Unknown view type: {view_type}")

                embeddings_list.append(batch_emb.cpu().numpy())
                labels_list.append(labels.numpy())
                ids_list.extend(img_ids)

        return (
            np.concatenate(embeddings_list, axis=0),
            np.concatenate(labels_list, axis=0),
            np.array(ids_list),
        )

    def extract_features(
        self, stream_name, split_name, debug_sample_size=None, load_cached_data=True
    ):
        """
        Main method to extract features for a specific stream and split.
        Handles caching, multi-view processing, and concatenation.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORK_DIR, exist_ok=True)

        # Resolve paths
        emb_path, lbl_path, id_path = self._get_paths(stream_name, split_name)

        # Check Cache
        if load_cached_data and os.path.exists(emb_path):
            print(
                f"[{stream_name} - {split_name}] Loading cached embeddings from {emb_path}"
            )
            embeddings = np.load(emb_path)

            # Load labels/ids if needed and available
            labels = None
            ids = None
            if lbl_path and os.path.exists(lbl_path):
                labels = np.load(lbl_path)
            if id_path and os.path.exists(id_path):
                ids = np.load(id_path)

            return embeddings, labels, ids

        print(f"[{stream_name} - {split_name}] Computing features...")

        # Load Model
        model = get_backbone(stream_name)
        model = model.to(self.device)

        # Define Views
        views = ["global", "standard", "robust"]
        view_embeddings = []

        final_labels = None
        final_ids = None

        metadata_path = self._get_metadata_path(split_name)

        for view in views:
            # print(f"  Processing view: {view}")

            # Setup Dataset and Loader
            transform = get_transforms(stream_name, view)
            dataset = DogDataset(
                metadata_path, transform=transform, debug_sample_size=debug_sample_size
            )

            # Batch size can be standard for global/standard, but maybe smaller for robust (5x memory)
            # However, A100 40GB is large enough for BS=32 even with 5 crops.
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Run Inference
            emb, lbl, ids = self._run_inference(model, loader, view)
            view_embeddings.append(emb)

            # Store labels/ids from the first view (they should be identical across views)
            if final_labels is None:
                final_labels = lbl
                final_ids = ids

        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()

        # Concatenate Views (Intra-Stream Early Fusion)
        # Shape: (N, D_global + D_standard + D_robust)
        concatenated_embeddings = np.concatenate(view_embeddings, axis=1)

        # Save to Cache
        print(f"  Saving embeddings to {emb_path}")
        np.save(emb_path, concatenated_embeddings)

        # Save Labels and IDs if they don't exist or we want to ensure consistency
        # We only save if the path is defined (e.g. test set has no label path)
        if lbl_path:
            if not os.path.exists(lbl_path) or not load_cached_data:
                np.save(lbl_path, final_labels)

        if id_path:
            if not os.path.exists(id_path) or not load_cached_data:
                np.save(id_path, final_ids)

        return concatenated_embeddings, final_labels, final_ids
