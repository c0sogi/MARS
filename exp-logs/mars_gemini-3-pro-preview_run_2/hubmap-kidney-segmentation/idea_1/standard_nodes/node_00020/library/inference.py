import os
import numpy as np
import pandas as pd
import torch
import rasterio
import scipy.signal
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import rle_encode, seed_everything
from library.model import FPN


class InferenceRunner:
    """
    Handles inference for the FTU Detection task.
    Performs sliding window prediction with Gaussian weighting and RLE encoding.
    """

    def __init__(self, checkpoint_path: str):
        """
        Args:
            checkpoint_path (str): Path to the trained model weights.
        """
        self.device = Config.DEVICE
        self.checkpoint_path = checkpoint_path

        # Initialize Model
        self.model = FPN(num_classes=Config.CLASSES)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing Transform (Same as validation)
        self.transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        # Gaussian Kernel for smoothing overlaps
        self.kernel = self._get_gaussian_kernel(Config.TILE_SIZE)

    def _load_weights(self):
        """Loads model weights from checkpoint."""
        if os.path.exists(self.checkpoint_path):
            print(f"Loading weights from {self.checkpoint_path}")
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Checkpoint {self.checkpoint_path} not found. Using random weights."
            )

    def _get_gaussian_kernel(self, size: int, sigma_scale: float = 8.0) -> np.ndarray:
        """
        Generates a 2D Gaussian kernel for weighting tile predictions.
        """
        sigma = size / sigma_scale
        x = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        gauss = np.exp(-0.5 * np.square(x) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / np.max(kernel)

    def predict_large_image(self, image_path: str) -> str:
        """
        Performs sliding window inference on a single large TIFF image.

        Args:
            image_path (str): Path to the TIFF image.

        Returns:
            str: RLE encoded binary mask.
        """
        # Open image to get dimensions
        with rasterio.open(image_path) as src:
            h_img, w_img = src.shape

            # Initialize accumulation buffers
            # Using float16 to save memory for very large images, though float32 is safer for precision
            prob_map = np.zeros((h_img, w_img), dtype=np.float32)
            weight_map = np.zeros((h_img, w_img), dtype=np.float32)

            # Define stride
            stride = int(Config.TILE_SIZE * (1 - Config.INFERENCE_OVERLAP))

            # Generate coordinates
            y_points = list(range(0, h_img, stride))
            x_points = list(range(0, w_img, stride))

            # Adjust last points to ensure coverage if needed (though padding handles edges)
            # Standard sliding window usually covers everything if we pad the read.

            for y in y_points:
                for x in x_points:
                    # 1. Read Tile
                    # Calculate window size (handle edges)
                    h_read = min(Config.TILE_SIZE, h_img - y)
                    w_read = min(Config.TILE_SIZE, w_img - x)

                    window = rasterio.windows.Window(x, y, w_read, h_read)
                    if src.count >= 3:
                        img_tile = src.read([1, 2, 3], window=window)  # (3, H, W)
                    else:
                        img_tile = src.read([1], window=window)
                        img_tile = np.repeat(img_tile, 3, axis=0)

                    # 2. Pad Tile if necessary
                    if h_read < Config.TILE_SIZE or w_read < Config.TILE_SIZE:
                        pad_h = Config.TILE_SIZE - h_read
                        pad_w = Config.TILE_SIZE - w_read
                        img_tile = np.pad(
                            img_tile,
                            ((0, 0), (0, pad_h), (0, pad_w)),
                            mode="constant",
                            constant_values=0,
                        )

                    # 3. Preprocess
                    # Transpose to (H, W, C) for Albumentations
                    img_tile = np.transpose(img_tile, (1, 2, 0)).astype(np.uint8)
                    augmented = self.transform(image=img_tile)
                    input_tensor = (
                        augmented["image"].unsqueeze(0).to(self.device)
                    )  # (1, 3, 1024, 1024)

                    # 4. Inference
                    with torch.no_grad():
                        # Mixed precision inference
                        with torch.cuda.amp.autocast():
                            logits = self.model(input_tensor)
                        probs = (
                            torch.sigmoid(logits).squeeze().cpu().numpy()
                        )  # (1024, 1024)

                    # 5. Accumulate with Gaussian Weighting
                    # Crop the prediction and kernel to the valid read area
                    valid_pred = probs[:h_read, :w_read]
                    valid_kernel = self.kernel[:h_read, :w_read]

                    prob_map[y : y + h_read, x : x + w_read] += (
                        valid_pred * valid_kernel
                    )
                    weight_map[y : y + h_read, x : x + w_read] += valid_kernel

        # 6. Normalize and Threshold
        # Avoid division by zero
        weight_map[weight_map == 0] = 1.0
        prob_map /= weight_map

        mask = prob_map > Config.THRESHOLD
        mask = mask.astype(np.uint8)

        # 7. Encode
        rle = rle_encode(mask)
        return rle

    def generate_submission(self):
        """
        Generates predictions for all test images and saves the submission CSV.
        """
        print("Starting inference on test set...")

        # Load Test Metadata
        test_metadata_path = os.path.join(Config.METADATA_DIR, "test_metadata.csv")
        if not os.path.exists(test_metadata_path):
            raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}")

        test_df = pd.read_csv(test_metadata_path)

        results = []

        for idx, row in test_df.iterrows():
            img_id = row["id"]
            img_rel_path = row["image_path"]
            img_full_path = os.path.join(Config.INPUT_DIR, img_rel_path)

            print(f"Processing {img_id}...")

            if not os.path.exists(img_full_path):
                print(f"Error: Image {img_full_path} not found. Skipping.")
                results.append({"id": img_id, "predicted": ""})
                continue

            try:
                rle_mask = self.predict_large_image(img_full_path)
                results.append({"id": img_id, "predicted": rle_mask})
            except Exception as e:
                print(f"Error processing {img_id}: {str(e)}")
                results.append({"id": img_id, "predicted": ""})

        # Save Submission
        submission_df = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_inference():
    """
    Main entry point for the inference module.
    """
    seed_everything(Config.SEED)

    # Path to the best model saved during training
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    runner = InferenceRunner(checkpoint_path)
    runner.generate_submission()
