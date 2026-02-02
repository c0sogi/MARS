import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.model import CoRes2NetUNet
from library.dataset import DenoisingDataset
from library.utils import create_submission_file, calculate_rmse, set_seed


class InferenceEngine:
    """
    Inference Engine for CoRes2Net-UNet.
    Handles model loading, tiled inference with TTA, validation, and submission generation.
    """

    def __init__(self, config: Config):
        """
        Initialize the Inference Engine.

        Args:
            config (Config): Configuration object containing paths and hyperparameters.
        """
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Initialize Model
        self.model = CoRes2NetUNet(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_filters=config.BASE_FILTERS,
        ).to(self.device)

        # Load Weights
        self._load_checkpoint()

    def _load_checkpoint(self):
        """
        Loads the model weights from the checkpoint path specified in config.
        """
        if os.path.exists(self.config.CHECKPOINT_PATH):
            state_dict = torch.load(
                self.config.CHECKPOINT_PATH, map_location=self.device
            )
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {self.config.CHECKPOINT_PATH}")
        else:
            print(
                f"Warning: Checkpoint not found at {self.config.CHECKPOINT_PATH}. Using random initialization."
            )

    def _apply_tta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 8 geometric transformations (D4 group) to the input batch.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Augmented tensor of shape [B*8, C, H, W].
        """
        out = []
        # 0: Identity
        out.append(x)
        # 1: Rot90
        out.append(torch.rot90(x, 1, [2, 3]))
        # 2: Rot180
        out.append(torch.rot90(x, 2, [2, 3]))
        # 3: Rot270
        out.append(torch.rot90(x, 3, [2, 3]))
        # 4: HFlip
        out.append(torch.flip(x, [3]))
        # 5: VFlip
        out.append(torch.flip(x, [2]))
        # 6: Transpose (Rot90 + Flip V)
        out.append(torch.flip(torch.rot90(x, 1, [2, 3]), [2]))
        # 7: Anti-Transpose (Rot90 + Flip H)
        out.append(torch.flip(torch.rot90(x, 1, [2, 3]), [3]))

        return torch.cat(out, dim=0)

    def _reverse_tta(self, x: torch.Tensor, original_batch_size: int) -> torch.Tensor:
        """
        Reverses the TTA transformations and averages the results.

        Args:
            x (torch.Tensor): Augmented output tensor of shape [B*8, C, H, W].
            original_batch_size (int): The original batch size B.

        Returns:
            torch.Tensor: Averaged tensor of shape [B, C, H, W].
        """
        # Split into the 8 groups
        chunks = torch.chunk(x, 8, dim=0)

        # Inverse transforms
        # 0: Identity
        c0 = chunks[0]
        # 1: Rot90 -> Rot270 (3)
        c1 = torch.rot90(chunks[1], 3, [2, 3])
        # 2: Rot180 -> Rot180 (2)
        c2 = torch.rot90(chunks[2], 2, [2, 3])
        # 3: Rot270 -> Rot90 (1)
        c3 = torch.rot90(chunks[3], 1, [2, 3])
        # 4: HFlip -> HFlip
        c4 = torch.flip(chunks[4], [3])
        # 5: VFlip -> VFlip
        c5 = torch.flip(chunks[5], [2])
        # 6: Transpose -> Transpose
        c6 = torch.rot90(torch.flip(chunks[6], [2]), 3, [2, 3])
        # 7: Anti-Transpose -> Anti-Transpose
        c7 = torch.rot90(torch.flip(chunks[7], [3]), 3, [2, 3])

        # Stack and mean
        stacked = torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=0)
        return torch.mean(stacked, dim=0)

    def predict_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs inference on a single image using Tiled Inference and TTA.

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W].

        Returns:
            torch.Tensor: Predicted clean image tensor of shape [1, C, H, W].
        """
        self.model.eval()

        b, c, h, w = image_tensor.shape
        patch_size = self.config.PATCH_SIZE
        overlap = self.config.OVERLAP_RATIO
        stride = int(patch_size * (1 - overlap))

        # Output containers
        output = torch.zeros((b, c, h, w), device=self.device)
        count = torch.zeros((b, c, h, w), device=self.device)

        # Calculate padding to ensure coverage
        pad_h = 0
        pad_w = 0
        if h < patch_size:
            pad_h = patch_size - h
        if w < patch_size:
            pad_w = patch_size - w

        # Pad image (Reflect padding handles boundaries gracefully)
        padded_img = torch.nn.functional.pad(
            image_tensor, (0, pad_w, 0, pad_h), mode="reflect"
        )
        _, _, h_pad, w_pad = padded_img.shape

        # Extract patches
        y_range = list(range(0, h_pad - patch_size + 1, stride))
        if (h_pad - patch_size) % stride != 0:
            y_range.append(h_pad - patch_size)

        x_range = list(range(0, w_pad - patch_size + 1, stride))
        if (w_pad - patch_size) % stride != 0:
            x_range.append(w_pad - patch_size)

        batch_patches = []
        batch_coords = []
        inference_batch_size = self.config.BATCH_SIZE

        # Adjust batch size if TTA is enabled (8x expansion)
        if self.config.TTA_ENABLED:
            inference_batch_size = max(1, inference_batch_size // 8)

        with torch.no_grad():
            for y in y_range:
                for x in x_range:
                    patch = padded_img[:, :, y : y + patch_size, x : x + patch_size]
                    batch_patches.append(patch)
                    batch_coords.append((y, x))

                    if len(batch_patches) >= inference_batch_size:
                        # Process Batch
                        inp = torch.cat(batch_patches, dim=0)  # [B, C, H, W]

                        if self.config.TTA_ENABLED:
                            inp_aug = self._apply_tta(inp)
                            pred_aug = self.model(inp_aug)
                            pred = self._reverse_tta(pred_aug, inp.shape[0])
                        else:
                            pred = self.model(inp)

                        # Accumulate
                        for i, (py, px) in enumerate(batch_coords):
                            output[
                                :, :, py : py + patch_size, px : px + patch_size
                            ] += pred[i : i + 1]
                            count[
                                :, :, py : py + patch_size, px : px + patch_size
                            ] += 1.0

                        batch_patches = []
                        batch_coords = []

            # Process remaining patches
            if len(batch_patches) > 0:
                inp = torch.cat(batch_patches, dim=0)
                if self.config.TTA_ENABLED:
                    inp_aug = self._apply_tta(inp)
                    pred_aug = self.model(inp_aug)
                    pred = self._reverse_tta(pred_aug, inp.shape[0])
                else:
                    pred = self.model(inp)

                for i, (py, px) in enumerate(batch_coords):
                    output[:, :, py : py + patch_size, px : px + patch_size] += pred[
                        i : i + 1
                    ]
                    count[:, :, py : py + patch_size, px : px + patch_size] += 1.0

        # Average overlapping regions
        noise_pred = output / count

        # Crop back to original size
        noise_pred = noise_pred[:, :, :h, :w]

        # Calculate Clean Image: Clean = Noisy - Noise
        clean_pred = image_tensor - noise_pred

        # Clip to valid range [0, 1]
        clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

        return clean_pred

    def validate(self):
        """
        Runs validation on the validation set and prints the RMSE metric.
        """
        print("Starting Validation...")
        val_dataset = DenoisingDataset("val", self.config, load_cached_data=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        rmse_list = []

        with torch.no_grad():
            for noisy, clean, img_id in val_loader:
                noisy = noisy.to(self.device)
                clean_target = clean.numpy()

                # Predict
                pred_clean = self.predict_image(noisy)
                pred_clean_np = pred_clean.cpu().numpy()

                # Calculate RMSE
                val_rmse = calculate_rmse(clean_target, pred_clean_np)
                rmse_list.append(val_rmse)

        mean_rmse = np.mean(rmse_list)
        print(f"Validation RMSE: {mean_rmse}")
        return mean_rmse

    def generate_submission(self):
        """
        Generates predictions for the test set and saves them to the submission CSV.
        """
        print("Generating Submission...")
        test_dataset = DenoisingDataset("test", self.config, load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        predictions = {}

        with torch.no_grad():
            for noisy, img_id_tuple in test_loader:
                img_id = img_id_tuple[0]
                noisy = noisy.to(self.device)

                # Predict
                pred_clean = self.predict_image(noisy)

                # Convert to numpy (Squeeze to remove batch and channel dims for 2D output)
                pred_clean_np = pred_clean.squeeze().cpu().numpy()

                predictions[img_id] = pred_clean_np

        create_submission_file(predictions, self.config.SUBMISSION_PATH)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
