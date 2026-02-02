import os
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import load_data
from library.utils import seed_everything


class FeatureExtractor:
    """
    Handles the extraction of features using Dual-Backbones (CNN + ViT)
    and aggregates them with tabular features.

    Implements augmentation-averaged embedding extraction to enforce
    rotational invariance.
    """

    def __init__(self):
        self.device = Config.DEVICE
        print(f"Initializing FeatureExtractor on {self.device}...")

        # Initialize CNN Backbone (ResNet50)
        # num_classes=0 removes the classification head
        # global_pool='avg' ensures we get the pooled feature vector
        self.cnn = timm.create_model(
            Config.BACKBONE_CNN, pretrained=True, num_classes=0, global_pool="avg"
        ).to(self.device)
        self.cnn.eval()

        # Initialize ViT Backbone (ViT-Base)
        # global_pool='token' uses the [CLS] token embedding
        self.vit = timm.create_model(
            Config.BACKBONE_VIT, pretrained=True, num_classes=0, global_pool="token"
        ).to(self.device)
        self.vit.eval()

    def _extract_from_dataset(self, dataset, desc="Dataset"):
        """
        Iterates over the dataset, generates 4-view averaged embeddings
        for both backbones, and collects tabular features/targets.

        Args:
            dataset (Dataset): The LeafDataset to iterate over.
            desc (str): Description for logging.

        Returns:
            dict: Dictionary containing numpy arrays of features and metadata.
        """
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        cnn_embeddings = []
        vit_embeddings = []
        tabular_features = []
        targets = []
        ids = []

        print(f"Starting feature extraction for {desc}...")

        with torch.no_grad():
            for batch_idx, (images, tab_feats, labels, img_ids) in enumerate(loader):
                # images shape: (B, 4, 3, H, W)
                # tab_feats shape: (B, 192)
                # labels shape: (B) (or -1 for test)
                # img_ids shape: (B)

                B, V, C, H, W = images.shape

                # Flatten views into batch dimension for inference
                # Input becomes (B*4, 3, H, W)
                flat_images = images.view(-1, C, H, W).to(self.device)

                # --- CNN Extraction ---
                # Output: (B*4, 2048)
                cnn_out = self.cnn(flat_images)
                # Reshape to (B, 4, 2048) and average across views
                cnn_out = cnn_out.view(B, V, -1).mean(dim=1)
                cnn_embeddings.append(cnn_out.cpu().numpy())

                # --- ViT Extraction ---
                # Output: (B*4, 768)
                vit_out = self.vit(flat_images)
                # Reshape to (B, 4, 768) and average across views
                vit_out = vit_out.view(B, V, -1).mean(dim=1)
                vit_embeddings.append(vit_out.cpu().numpy())

                # --- Tabular & Meta ---
                tabular_features.append(tab_feats.numpy())
                targets.append(labels.numpy())
                ids.append(img_ids.numpy())

        # Concatenate all batches
        print(f"Finished extraction for {desc}.")
        return {
            "cnn": np.vstack(cnn_embeddings),
            "vit": np.vstack(vit_embeddings),
            "tabular": np.vstack(tabular_features),
            "targets": np.concatenate(targets),
            "ids": np.concatenate(ids),
        }

    def run(self, load_cached_data=True):
        """
        Main execution method. Loads data, checks cache, extracts features if needed,
        and returns dictionary of arrays.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            dict: Nested dictionary containing train/val/test features.
        """
        seed_everything(Config.SEED)
        Config.setup()  # Ensure directories exist

        # Define cache paths mapping
        cache_files = {
            "train": {
                "cnn": Config.get_cache_path(Config.CACHE_TRAIN_CNN),
                "vit": Config.get_cache_path(Config.CACHE_TRAIN_VIT),
                "tab": Config.get_cache_path(Config.CACHE_TRAIN_TAB),
                "tgt": Config.get_cache_path(Config.CACHE_TRAIN_TARGETS),
            },
            "val": {
                "cnn": Config.get_cache_path(Config.CACHE_VAL_CNN),
                "vit": Config.get_cache_path(Config.CACHE_VAL_VIT),
                "tab": Config.get_cache_path(Config.CACHE_VAL_TAB),
                "tgt": Config.get_cache_path(Config.CACHE_VAL_TARGETS),
            },
            "test": {
                "cnn": Config.get_cache_path(Config.CACHE_TEST_CNN),
                "vit": Config.get_cache_path(Config.CACHE_TEST_VIT),
                "tab": Config.get_cache_path(Config.CACHE_TEST_TAB),
                "ids": Config.get_cache_path(Config.CACHE_TEST_IDS),
            },
        }

        # Check if all cache files exist
        all_cached = True
        if load_cached_data:
            for split in cache_files:
                for key, path in cache_files[split].items():
                    if not os.path.exists(path):
                        print(f"Cache miss: {path}")
                        all_cached = False
                        break
                if not all_cached:
                    break
        else:
            all_cached = False

        # Return cached data if available
        if all_cached:
            print("Loading features from cache...")
            data = {}
            for split in ["train", "val", "test"]:
                data[split] = {}
                for key, path in cache_files[split].items():
                    data[split][key] = np.load(path)
            return data

        # If not cached, compute
        print("Cache not found or forced reload. Extracting features...")

        # Load Raw Data
        # We pass debug flag based on Config to limit dataset if needed
        train_ds, val_ds, test_ds, _ = load_data(
            debug=(Config.DEBUG_SAMPLE_SIZE is not None)
        )

        # Extract
        train_data = self._extract_from_dataset(train_ds, "Train")
        val_data = self._extract_from_dataset(val_ds, "Validation")
        test_data = self._extract_from_dataset(test_ds, "Test")

        # Save to Cache
        print("Saving features to cache...")

        # Train
        np.save(cache_files["train"]["cnn"], train_data["cnn"])
        np.save(cache_files["train"]["vit"], train_data["vit"])
        np.save(cache_files["train"]["tab"], train_data["tabular"])
        np.save(cache_files["train"]["tgt"], train_data["targets"])

        # Val
        np.save(cache_files["val"]["cnn"], val_data["cnn"])
        np.save(cache_files["val"]["vit"], val_data["vit"])
        np.save(cache_files["val"]["tab"], val_data["tabular"])
        np.save(cache_files["val"]["tgt"], val_data["targets"])

        # Test
        np.save(cache_files["test"]["cnn"], test_data["cnn"])
        np.save(cache_files["test"]["vit"], test_data["vit"])
        np.save(cache_files["test"]["tab"], test_data["tabular"])
        np.save(cache_files["test"]["ids"], test_data["ids"])

        # Construct return dict
        # Note: We map the internal keys to the expected output keys
        return {
            "train": {
                "cnn": train_data["cnn"],
                "vit": train_data["vit"],
                "tab": train_data["tabular"],
                "tgt": train_data["targets"],
            },
            "val": {
                "cnn": val_data["cnn"],
                "vit": val_data["vit"],
                "tab": val_data["tabular"],
                "tgt": val_data["targets"],
            },
            "test": {
                "cnn": test_data["cnn"],
                "vit": test_data["vit"],
                "tab": test_data["tabular"],
                "ids": test_data["ids"],
            },
        }
