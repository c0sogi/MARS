import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.architecture import ResDnCNN
from library.data_loader import TestDataset


class GeometricTTA:
    """
    Implements Geometric Test-Time Augmentation (TTA).
    Performs 8 transforms (D4 group): 4 Rotations x 2 Flips.
    """

    def __init__(self, device):
        self.device = device

    def apply_aug(self, x, k, flip):
        """
        Applies augmentation: Flip (optional) -> Rotate 90*k
        x: Tensor [B, C, H, W]
        k: int (0, 1, 2, 3) representing 90 degree rotations
        flip: bool, whether to flip horizontally
        """
        # 1. Flip Horizontal (dim 3 in NCHW)
        if flip:
            x = torch.flip(x, dims=[3])

        # 2. Rotate (dims 2, 3 are H, W)
        if k > 0:
            x = torch.rot90(x, k=k, dims=[2, 3])

        return x

    def reverse_aug(self, x, k, flip):
        """
        Reverses the augmentation: Rotate -90*k -> Flip (optional)
        Note: The reverse order of operations is required.
        Inverse of (Rot * Flip) is (Flip_inv * Rot_inv).
        Since Flip_inv = Flip, we do: Flip(Rot(-k, x))
        """
        # 1. Reverse Rotation
        if k > 0:
            x = torch.rot90(x, k=-k, dims=[2, 3])

        # 2. Reverse Flip
        if flip:
            x = torch.flip(x, dims=[3])

        return x

    def predict_with_tta(self, model, x):
        """
        Runs inference with 8-fold TTA.
        Returns the averaged prediction tensor.
        """
        preds = []

        # Iterate through all 8 combinations
        for flip in [False, True]:
            for k in [0, 1, 2, 3]:
                # Transform input
                x_aug = self.apply_aug(x, k, flip)

                # Predict (Model predicts residual noise)
                with torch.no_grad():
                    y_aug = model(x_aug)

                # Inverse transform prediction
                y_pred = self.reverse_aug(y_aug, k, flip)

                preds.append(y_pred)

        # Stack and average
        preds = torch.stack(preds, dim=0)
        avg_pred = torch.mean(preds, dim=0)
        return avg_pred


class EnsemblePredictor:
    """
    Manages loading of ensemble models and aggregating their predictions.
    """

    def __init__(self, device=None):
        self.device = device if device else Config.DEVICE
        self.models = []
        self.tta = GeometricTTA(self.device)
        self._load_models()

    def _load_models(self):
        """Loads all available model checkpoints from the working directory."""
        print(f"Loading ensemble models from {Config.WORKING_DIR}...")

        for i in range(Config.NUM_ENSEMBLE_MODELS):
            model_path = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")

            if not os.path.exists(model_path):
                print(f"Warning: Checkpoint {model_path} not found. Skipping.")
                continue

            # Initialize model architecture
            model = ResDnCNN(
                depth=Config.MODEL_DEPTH,
                filters=Config.MODEL_FILTERS,
                input_channels=Config.INPUT_CHANNELS,
                output_channels=Config.OUTPUT_CHANNELS,
            )

            # Load weights
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"Successfully loaded model_{i}")
            except Exception as e:
                print(f"Error loading {model_path}: {e}")

        if not self.models:
            print("CRITICAL: No models loaded. Inference will fail or produce zeros.")

    def predict(self, image_tensor):
        """
        Predicts the noise residual for a single image using the ensemble.
        image_tensor: [1, C, H, W]
        """
        if not self.models:
            return torch.zeros_like(image_tensor)

        model_preds = []
        for model in self.models:
            # Get TTA averaged prediction for this model
            pred = self.tta.predict_with_tta(model, image_tensor)
            model_preds.append(pred)

        # Average across ensemble members
        ensemble_pred = torch.stack(model_preds, dim=0)
        final_noise = torch.mean(ensemble_pred, dim=0)

        return final_noise


def generate_submission():
    """
    Main inference routine.
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Starting submission generation...")

    # Setup
    device = Config.DEVICE
    predictor = EnsemblePredictor(device)

    # Data Loader
    # TestDataset returns (image_tensor, image_id)
    # Batch size 1 is necessary because test images may have varying dimensions
    test_dataset = TestDataset(Config.TEST_METADATA_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    results_id = []
    results_val = []

    print(f"Processing {len(test_dataset)} test images...")

    for i, (noisy_img, img_id_tuple) in enumerate(test_loader):
        # Unwrap tuple from dataloader (batch_size=1)
        img_id_str = img_id_tuple[0]

        # Move to device
        noisy_img = noisy_img.to(device)

        # Predict Noise
        predicted_noise = predictor.predict(noisy_img)

        # Reconstruct Clean Image: Input - Noise
        clean_img = noisy_img - predicted_noise

        # Clip to valid range [0, 1]
        clean_img = torch.clamp(clean_img, 0.0, 1.0)

        # Move to CPU numpy
        # Shape is [1, 1, H, W] -> squeeze to [H, W]
        clean_numpy = clean_img.squeeze().cpu().numpy()

        # Flatten and format for submission
        h, w = clean_numpy.shape

        # Parse image ID (remove extension if present, though metadata usually has full filename)
        # Task description example: '110_1_1' for image '110.png'
        base_id = os.path.splitext(img_id_str)[0]

        # Create coordinate grids (1-based indexing)
        # We iterate to create the list. For 540x420 ~ 220k pixels, this is fast enough.
        # Vectorized string creation can be tricky with mixed types, list comp is safe.

        # Flatten image in row-major order (default for numpy flatten)
        flat_pixels = clean_numpy.flatten()

        # Generate IDs
        # Row indices: repeat 1..H, W times each? No, flatten is row-major.
        # Row 1: (1,1), (1,2)... (1,W)
        # Row 2: (2,1)...

        # Using list comprehension for clarity and correctness
        # This matches the order of flat_pixels
        current_ids = [
            f"{base_id}_{r}_{c}" for r in range(1, h + 1) for c in range(1, w + 1)
        ]

        results_id.extend(current_ids)
        results_val.extend(flat_pixels)

    # Create DataFrame
    print("Constructing submission DataFrame...")
    df_sub = pd.DataFrame({"id": results_id, "value": results_val})

    # Save
    save_path = Config.SUBMISSION_PATH
    print(f"Saving submission to {save_path}...")
    df_sub.to_csv(save_path, index=False)
    print("Submission generation complete.")
