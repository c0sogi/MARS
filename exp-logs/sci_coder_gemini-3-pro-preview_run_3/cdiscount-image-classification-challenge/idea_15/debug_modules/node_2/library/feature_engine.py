import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from library.config import Config
from library.utils import seed_everything
from library.bson_loader import get_bson_loader


class FeatureEngine:
    """
    Handles the extraction of visual features using a Dual-Backbone architecture
    (ResNet50 + EfficientNet-B0). Features are aggregated per product and cached
    to disk to decouple feature extraction from classifier training.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.img_size = Config.RESIZE_SIZE
        seed_everything(Config.SEED)

    def _get_transforms(self):
        """Returns the preprocessing transforms required for the backbones."""
        return transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.MEAN, std=Config.STD),
            ]
        )

    def _build_feature_extractor(self):
        """
        Initializes the dual backbone model in evaluation mode.
        Returns a tuple of (model1, model2).
        """
        # Backbone 1: ResNet50 (2048 dim)
        resnet = models.resnet50(weights="DEFAULT")
        resnet.fc = nn.Identity()  # Remove classification head
        resnet.to(self.device)
        resnet.eval()

        # Backbone 2: EfficientNet-B0 (1280 dim)
        effnet = models.efficientnet_b0(weights="DEFAULT")
        effnet.classifier = nn.Identity()  # Remove classification head
        effnet.to(self.device)
        effnet.eval()

        return resnet, effnet

    def _process_split(
        self,
        metadata_path,
        bson_path,
        output_feat_path,
        output_label_path,
        output_id_path=None,
        mode="train",
    ):
        """
        Runs the feature extraction loop for a specific data split.
        """
        print(f"Starting feature extraction for {mode} split...")
        print(f"Source: {bson_path}")
        print(f"Metadata: {metadata_path}")

        # Setup
        transform = self._get_transforms()
        loader = get_bson_loader(
            metadata_path=metadata_path,
            bson_path=bson_path,
            batch_size=Config.EXTRACT_BATCH_SIZE,
            transform=transform,
            mode=mode,
            num_workers=Config.NUM_WORKERS,
            shuffle=False,  # Order must be preserved for correspondence with labels/IDs
        )

        model1, model2 = self._build_feature_extractor()

        all_features = []
        all_labels = []
        all_ids = []

        # Inference Loop
        with torch.no_grad():
            for batch_idx, (flat_images, flat_ids, flat_labels, sizes) in enumerate(
                loader
            ):
                flat_images = flat_images.to(self.device)

                # 1. Extract Features from both backbones
                # ResNet output: (N_imgs, 2048)
                feats1 = model1(flat_images)
                # EfficientNet output: (N_imgs, 1280)
                feats2 = model2(flat_images)

                # 2. Concatenate Features: (N_imgs, 3328)
                feats_concat = torch.cat([feats1, feats2], dim=1)

                # 3. Aggregate per product (Mean Pooling)
                # We need to split the flattened batch back into product groups
                # sizes is a list of [n_imgs_prod1, n_imgs_prod2, ...]
                feats_split = torch.split(feats_concat, sizes)

                # Stack the means: (Batch_Size, 3328)
                batch_feats_agg = torch.stack([f.mean(dim=0) for f in feats_split])

                # 4. Collect results
                all_features.append(batch_feats_agg.cpu().numpy())

                # For labels/ids, we need one per product.
                # The loader returns flat_labels/flat_ids which repeats values per image.
                # We need to extract unique values per product (which corresponds to the 'sizes' structure)

                # Reconstruct product-level IDs and Labels
                # Since collate_fn repeats them, we can just take the value at the start index of each split
                # But simpler: the loader iterates linearly. We can just accumulate the raw values from the dataset?
                # No, the loader provides the ground truth for the current batch.

                # Efficient way to get 1 label per product from flat tensors:
                # We can use a cumulative sum on sizes to find indices, or just process the list logic.
                # Since flat_ids/labels are constructed as [id1]*n1 + [id2]*n2...
                # We can construct a mask or just iterate. Given Python loop overhead is low for batch size 512:

                cursor = 0
                batch_p_ids = []
                batch_p_labels = []

                # Move to CPU for list processing
                f_ids_cpu = flat_ids.numpy()
                f_lbls_cpu = flat_labels.numpy()

                for s in sizes:
                    # Take the first item of the group
                    batch_p_ids.append(f_ids_cpu[cursor])
                    batch_p_labels.append(f_lbls_cpu[cursor])
                    cursor += s

                all_ids.append(np.array(batch_p_ids, dtype=np.int64))
                all_labels.append(np.array(batch_p_labels, dtype=np.int64))

                if batch_idx % 100 == 0:
                    print(f"Processed batch {batch_idx}/{len(loader)}")

        # Concatenate all batches
        print("Concatenating arrays...")
        final_features = np.concatenate(all_features, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        final_ids = np.concatenate(all_ids, axis=0)

        print(f"Saving features shape {final_features.shape} to {output_feat_path}")
        np.save(output_feat_path, final_features)

        if output_label_path:
            print(f"Saving labels shape {final_labels.shape} to {output_label_path}")
            np.save(output_label_path, final_labels)

        if output_id_path:
            print(f"Saving ids shape {final_ids.shape} to {output_id_path}")
            np.save(output_id_path, final_ids)

        # Clear memory
        del (
            final_features,
            final_labels,
            final_ids,
            all_features,
            all_labels,
            all_ids,
            model1,
            model2,
        )
        torch.cuda.empty_cache()

    def extract_features(self, load_cached_data=True):
        """
        Main entry point. Checks cache and runs extraction for Train, Val, and Test splits.
        """
        Config.make_dirs()

        # Define tasks
        tasks = [
            {
                "mode": "train",
                "meta": Config.TRAIN_META,
                "bson": Config.TRAIN_BSON,
                "out_feat": Config.TRAIN_FEATURES,
                "out_label": Config.TRAIN_LABELS,
                "out_id": None,  # Train IDs not strictly needed for training, but labels are
            },
            {
                "mode": "val",
                "meta": Config.VAL_META,
                "bson": Config.TRAIN_BSON,
                "out_feat": Config.VAL_FEATURES,
                "out_label": Config.VAL_LABELS,
                "out_id": None,
            },
            {
                "mode": "test",
                "meta": Config.TEST_META,
                "bson": Config.TEST_BSON,
                "out_feat": Config.TEST_FEATURES,
                "out_label": None,  # No labels for test
                "out_id": Config.TEST_IDS,  # Need IDs for submission
            },
        ]

        for task in tasks:
            # Check cache
            files_to_check = [task["out_feat"]]
            if task["out_label"]:
                files_to_check.append(task["out_label"])
            if task["out_id"]:
                files_to_check.append(task["out_id"])

            cache_exists = all(os.path.exists(f) for f in files_to_check)

            if load_cached_data and cache_exists:
                print(f"Cache found for {task['mode']} split. Skipping extraction.")
                continue

            # Run extraction
            self._process_split(
                metadata_path=task["meta"],
                bson_path=task["bson"],
                output_feat_path=task["out_feat"],
                output_label_path=task["out_label"],
                output_id_path=task["out_id"],
                mode=task["mode"],
            )

        print("Feature extraction complete.")
