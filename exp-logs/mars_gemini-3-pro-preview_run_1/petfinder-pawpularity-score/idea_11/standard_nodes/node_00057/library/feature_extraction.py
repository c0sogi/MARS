import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, SiglipModel
from library.config import Config
from library.dataset import get_dataloader
from library.utils import seed_everything


class DeepFeatureExtractor:
    """
    Handles the extraction of deep learning features using pre-trained backbones.
    Implements caching, feature-space augmentation (flip averaging), and
    supports SigLIP, DINOv2, and ConvNeXt architectures.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        seed_everything(Config.SEED)

    def _get_model_and_processor(self, backbone_key):
        """
        Loads the specific model and processor based on the backbone key.
        """
        cfg = Config.BACKBONES[backbone_key]
        model_id = cfg["model_id"]

        print(f"Loading {model_id} ({cfg['description']})...")

        processor = AutoImageProcessor.from_pretrained(model_id)

        # SigLIP requires specific handling for image feature extraction
        if "siglip" in backbone_key:
            model = SiglipModel.from_pretrained(model_id)
        else:
            model = AutoModel.from_pretrained(model_id)

        model.to(self.device)
        model.eval()

        return model, processor

    def _forward_pass(self, model, pixel_values, backbone_key):
        """
        Performs a forward pass to extract embeddings based on architecture.
        """
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            if "siglip" in backbone_key:
                # SigLIP: use get_image_features for normalized embeddings
                features = model.get_image_features(pixel_values)
            elif "dinov2" in backbone_key:
                # DINOv2: Take the CLS token (index 0) from the last hidden state
                outputs = model(pixel_values)
                features = outputs.last_hidden_state[:, 0, :]
            elif "convnext" in backbone_key:
                # ConvNeXt: Use pooler_output (Global Average Pooling)
                outputs = model(pixel_values)
                features = outputs.pooler_output
            else:
                raise ValueError(f"Unknown backbone key: {backbone_key}")

        return features

    def _process_dataset(self, dataframe, backbone_key, batch_size=32, is_train=False):
        """
        Iterates through the dataset, extracting features with flip averaging.
        """
        model, processor = self._get_model_and_processor(backbone_key)

        dataloader = get_dataloader(
            dataframe=dataframe,
            processor=processor,
            batch_size=batch_size,
            is_train=is_train,
            return_flipped=True,  # Enable Feature-Space Augmentation
            shuffle=False,
        )

        all_features = []
        all_meta = []
        all_ids = []
        all_targets = []

        print(f"Extracting features for {backbone_key}...")

        for batch in tqdm(dataloader, desc=f"Inference {backbone_key}"):
            # 1. Extract features for original images
            features_orig = self._forward_pass(
                model, batch["pixel_values"], backbone_key
            )

            # 2. Extract features for flipped images
            features_flip = self._forward_pass(
                model, batch["pixel_values_flipped"], backbone_key
            )

            # 3. Feature-Space Augmentation: Average the embeddings
            features_avg = (features_orig + features_flip) / 2.0

            # Move to CPU and collect
            all_features.append(features_avg.cpu().numpy())
            all_meta.append(batch["features"].numpy())
            all_ids.extend(batch["Id"])

            if is_train and "label" in batch:
                all_targets.append(batch["label"].numpy())

        # Cleanup model to free GPU memory
        del model
        del processor
        torch.cuda.empty_cache()
        gc.collect()

        # Concatenate results
        result = {
            "features": np.vstack(all_features),
            "meta": np.vstack(all_meta),
            "ids": np.array(all_ids),
        }

        if is_train and all_targets:
            result["targets"] = np.concatenate(all_targets)

        return result

    def extract_features(
        self, dataframe, backbone_key, subset_name, load_cached_data=True
    ):
        """
        Main method to get features. Checks cache first, otherwise computes and saves.

        Args:
            dataframe (pd.DataFrame): Data to process.
            backbone_key (str): Key from Config.BACKBONES (e.g., 'siglip').
            subset_name (str): Identifier for the subset (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            dict: Dictionary containing 'features', 'meta', 'ids', and optionally 'targets'.
        """
        # Define cache file paths
        prefix = f"{backbone_key}_{subset_name}"
        paths = {
            "features": os.path.join(self.cache_dir, f"{prefix}_features.npy"),
            "meta": os.path.join(self.cache_dir, f"{prefix}_meta.npy"),
            "ids": os.path.join(self.cache_dir, f"{prefix}_ids.npy"),
            "targets": os.path.join(self.cache_dir, f"{prefix}_targets.npy"),
        }

        # Determine if we expect targets based on subset name or dataframe columns
        expect_targets = "Pawpularity" in dataframe.columns

        # 1. Try to load from cache
        if load_cached_data:
            try:
                print(f"Attempting to load cached data for {prefix}...")
                data = {
                    "features": np.load(paths["features"]),
                    "meta": np.load(paths["meta"]),
                    "ids": np.load(paths["ids"]),
                }
                if expect_targets:
                    if os.path.exists(paths["targets"]):
                        data["targets"] = np.load(paths["targets"])
                    else:
                        raise FileNotFoundError("Targets file missing from cache.")

                print("Cache hit. Data loaded successfully.")
                return data
            except (FileNotFoundError, OSError, ValueError) as e:
                print(f"Cache miss or error ({e}). Proceeding to extraction.")

        # 2. Compute from scratch
        batch_size = Config.BACKBONES[backbone_key]["batch_size"]

        # If debugging, reduce dataframe size
        if Config.DEBUG:
            print(
                f"DEBUG MODE: Reducing {subset_name} size to {Config.DEBUG_SAMPLE_SIZE}"
            )
            dataframe = dataframe.head(Config.DEBUG_SAMPLE_SIZE)

        extracted_data = self._process_dataset(
            dataframe, backbone_key, batch_size=batch_size, is_train=expect_targets
        )

        # 3. Save to cache
        print(f"Saving extracted data for {prefix} to {self.cache_dir}...")
        np.save(paths["features"], extracted_data["features"])
        np.save(paths["meta"], extracted_data["meta"])
        np.save(paths["ids"], extracted_data["ids"])

        if "targets" in extracted_data:
            np.save(paths["targets"], extracted_data["targets"])

        return extracted_data

    def run_all(self, load_cached_data=True):
        """
        Convenience method to extract features for all backbones and all splits.
        """
        # Load metadata
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        results = {}

        for backbone in Config.BACKBONES.keys():
            print(f"\n{'='*20}\nProcessing Backbone: {backbone}\n{'='*20}")

            results[f"{backbone}_train"] = self.extract_features(
                train_df, backbone, "train", load_cached_data
            )
            results[f"{backbone}_val"] = self.extract_features(
                val_df, backbone, "val", load_cached_data
            )
            results[f"{backbone}_test"] = self.extract_features(
                test_df, backbone, "test", load_cached_data
            )

        return results
