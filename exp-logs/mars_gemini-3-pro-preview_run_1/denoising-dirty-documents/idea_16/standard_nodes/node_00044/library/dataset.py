import torch
import numpy as np
from torch.utils.data import Dataset
from library.config import AUGMENTATION_PARAMS, INFERENCE_PADDING_MULTIPLE
from library.utils import load_dataset_images


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for the Denoising Task.

    Features:
    - Loads paired noisy/clean images using a caching mechanism.
    - Implements Dual-Scale cropping (Stream A vs Stream B).
    - Applies Reflection Padding for boundary handling.
    - Performs geometric augmentations (Flip, Rotate) during training.
    - Handles padding for inference to ensure dimensions match model requirements.
    """

    def __init__(
        self,
        metadata_df,
        img_size=None,
        augment=False,
        cache_name="dataset_cache",
        load_cached_data=True,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id', 'noisy_image_path', etc.
            img_size (tuple, optional): Target crop size (H, W) for training.
                                        If None, full images are used (padded for inference).
            augment (bool): Whether to apply geometric augmentations.
            cache_name (str): Identifier for the cache file.
            load_cached_data (bool): Whether to attempt loading from existing cache.
        """
        self.img_size = img_size
        self.augment = augment

        # Load images into memory using the utility function (handles caching)
        # Returns dictionaries: {id: np.array}
        self.noisy_imgs, self.clean_imgs = load_dataset_images(
            metadata_df, cache_name=cache_name, load_cached_data=load_cached_data
        )

        # Create a sorted list of IDs to ensure deterministic iteration order
        self.ids = sorted(list(self.noisy_imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Retrieve noisy image
        noisy = self.noisy_imgs[img_id]

        # Retrieve clean image if available (Train/Val), else create dummy (Test)
        if img_id in self.clean_imgs:
            clean = self.clean_imgs[img_id]
        else:
            # Create a placeholder clean image of the same shape for the test set
            clean = np.zeros_like(noisy)

        # Apply specific processing based on mode
        if self.augment and self.img_size is not None:
            noisy, clean = self._process_train(noisy, clean)
        else:
            noisy, clean = self._process_inference(noisy, clean)

        # Convert to PyTorch Tensors
        # Shape: (H, W) -> (1, H, W)
        # Type: float32
        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
        clean_t = torch.from_numpy(clean).float().unsqueeze(0)

        return noisy_t, clean_t, img_id

    def _process_train(self, noisy, clean):
        """
        Applies random cropping and geometric augmentations for training.
        """
        h, w = noisy.shape
        target_h, target_w = self.img_size

        # 1. Pad if image is smaller than the target crop size
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        if pad_h > 0 or pad_w > 0:
            # Symmetric reflection padding
            pt = pad_h // 2
            pb = pad_h - pt
            pl = pad_w // 2
            pr = pad_w - pl

            noisy = np.pad(noisy, ((pt, pb), (pl, pr)), mode="reflect")
            clean = np.pad(clean, ((pt, pb), (pl, pr)), mode="reflect")

        # 2. Random Crop
        # Update dimensions after potentially padding
        h_curr, w_curr = noisy.shape

        # Select random top-left corner
        # If dimensions match exactly, range is [0, 1), so index is 0
        top = np.random.randint(0, h_curr - target_h + 1)
        left = np.random.randint(0, w_curr - target_w + 1)

        noisy = noisy[top : top + target_h, left : left + target_w]
        clean = clean[top : top + target_h, left : left + target_w]

        # 3. Geometric Augmentations
        # Horizontal Flip
        if np.random.rand() < AUGMENTATION_PARAMS.get("horizontal_flip_prob", 0.5):
            noisy = np.fliplr(noisy)
            clean = np.fliplr(clean)

        # Vertical Flip
        if np.random.rand() < AUGMENTATION_PARAMS.get("vertical_flip_prob", 0.5):
            noisy = np.flipud(noisy)
            clean = np.flipud(clean)

        # Rotate 90 degrees (Randomly 1, 2, or 3 times)
        if np.random.rand() < AUGMENTATION_PARAMS.get("rotate90_prob", 0.5):
            k = np.random.randint(1, 4)
            noisy = np.rot90(noisy, k)
            clean = np.rot90(clean, k)

        # Ensure arrays are contiguous in memory after flips/rotations
        # This prevents negative stride errors when converting to Torch tensors
        return np.ascontiguousarray(noisy), np.ascontiguousarray(clean)

    def _process_inference(self, noisy, clean):
        """
        Applies padding to ensure dimensions are multiples of the network's downsampling factor.
        Used for Validation and Test sets.
        """
        h, w = noisy.shape
        factor = INFERENCE_PADDING_MULTIPLE

        # Calculate target dimensions (round up to nearest multiple)
        target_h = (h + factor - 1) // factor * factor
        target_w = (w + factor - 1) // factor * factor

        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            # Symmetric reflection padding
            pt = pad_h // 2
            pb = pad_h - pt
            pl = pad_w // 2
            pr = pad_w - pl

            noisy = np.pad(noisy, ((pt, pb), (pl, pr)), mode="reflect")
            clean = np.pad(clean, ((pt, pb), (pl, pr)), mode="reflect")

        return np.ascontiguousarray(noisy), np.ascontiguousarray(clean)
