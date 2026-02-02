import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import library.config as config

# Ensure deterministic behavior
config.seed_everything()


class DualStageFeatureExtractor(nn.Module):
    """
    Wrapper around ConvNeXt-Large to extract features from:
    - Stage 4 (Final Semantic): Native Global Average Pooling + LayerNorm
    - Stage 3 (Texture): Intermediate Feature Map + Global Average Pooling
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

        # Load pre-trained ConvNeXt Large
        # Using the string identifier as per config/torchvision
        weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        self.backbone = torchvision.models.convnext_large(weights=weights)
        self.backbone.to(self.device)
        self.backbone.eval()

        # Placeholder for the intermediate feature
        self.stage3_features = None

        # Register forward hook on Stage 3 (features[5])
        # ConvNeXt structure:
        # features[5] is the sequence of blocks for Stage 3
        self.hook_handle = self.backbone.features[5].register_forward_hook(
            self._stage3_hook
        )

    def _stage3_hook(self, module, input, output):
        # output shape: (Batch, Channels, Height, Width)
        # Perform Global Average Pooling immediately
        self.stage3_features = output.mean(dim=[-2, -1])

    def forward(self, x):
        # 1. Run through the feature extractor (Stages 0 to 7)
        # This triggers the hook at Stage 3
        x = self.backbone.features(x)

        # 2. Process Stage 4 (Final) features using native head components
        # ConvNeXt classifier head structure in torchvision:
        # classifier[0]: LayerNorm2d
        # classifier[1]: Flatten
        # classifier[2]: Linear (we skip this)

        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x)
        x = self.backbone.classifier[1](x)

        stage4_emb = x
        stage3_emb = self.stage3_features

        return stage3_emb, stage4_emb

    def __del__(self):
        if hasattr(self, "hook_handle"):
            self.hook_handle.remove()


def run_inference(dataloader, dataset_name, view_name, device, load_cached_data=True):
    """
    Runs inference to extract Stage 3 and Stage 4 features with TTA (Horizontal Flip).
    Implements caching mechanism to save/load features from disk.

    Args:
        dataloader: PyTorch DataLoader
        dataset_name (str): 'train', 'val', or 'test'
        view_name (str): 'global', 'standard', or 'local'
        device (str): 'cuda' or 'cpu'
        load_cached_data (bool): Whether to attempt loading from cache

    Returns:
        tuple: (stage3_features, stage4_features, ids, labels)
    """
    # Construct cache file paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    file_prefix = f"{dataset_name}_{view_name}"
    s3_path = os.path.join(cache_dir, f"{file_prefix}_stage3.npy")
    s4_path = os.path.join(cache_dir, f"{file_prefix}_stage4.npy")
    ids_path = os.path.join(cache_dir, f"{file_prefix}_ids.npy")
    labels_path = os.path.join(cache_dir, f"{file_prefix}_labels.npy")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(s3_path)
            and os.path.exists(s4_path)
            and os.path.exists(ids_path)
            and os.path.exists(labels_path)
        ):
            print(f"Loading cached features for {dataset_name} ({view_name})...")
            return (
                np.load(s3_path),
                np.load(s4_path),
                np.load(ids_path),
                np.load(labels_path),
            )

    print(f"Extracting features for {dataset_name} ({view_name})...")

    # Initialize model
    model = DualStageFeatureExtractor(device)

    s3_list = []
    s4_list = []
    ids_list = []
    labels_list = []

    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            images = images.to(device)

            # TTA: Create horizontally flipped version
            images_flip = torch.flip(images, dims=[3])

            # Forward pass - Original
            s3_orig, s4_orig = model(images)

            # Forward pass - Flipped
            s3_flip, s4_flip = model(images_flip)

            # Average embeddings (TTA)
            s3_avg = (s3_orig + s3_flip) / 2.0
            s4_avg = (s4_orig + s4_flip) / 2.0

            # Move to CPU and store
            s3_list.append(s3_avg.cpu().numpy())
            s4_list.append(s4_avg.cpu().numpy())

            ids_list.extend(image_ids)

            # Handle targets (might be tensor or list)
            if isinstance(targets, torch.Tensor):
                labels_list.extend(targets.cpu().numpy())
            else:
                labels_list.extend(targets)

    # Concatenate all batches
    s3_arr = np.vstack(s3_list)
    s4_arr = np.vstack(s4_list)
    ids_arr = np.array(ids_list)
    labels_arr = np.array(labels_list)

    # Save to cache
    print(f"Saving features to {cache_dir}...")
    np.save(s3_path, s3_arr)
    np.save(s4_path, s4_arr)
    np.save(ids_path, ids_arr)
    np.save(labels_path, labels_arr)

    return s3_arr, s4_arr, ids_arr, labels_arr
