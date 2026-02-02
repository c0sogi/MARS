import os
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.signal
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, rle_encode
from library.model import MultiTaskResNetFPN
from library.data import HuBMAPDataset, get_transforms


class InferenceRunner:
    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)
        self.device = torch.device(self.config.DEVICE)

        # Initialize Model
        self.model = MultiTaskResNetFPN()
        self.model.to(self.device)

        # Load Weights
        if os.path.exists(self.config.MODEL_PATH):
            state_dict = torch.load(self.config.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model weights from {self.config.MODEL_PATH}")
        else:
            raise FileNotFoundError(
                f"Model weights not found at {self.config.MODEL_PATH}"
            )

        self.model.eval()

    def _get_gaussian_kernel(self, size):
        """Generates a 2D Gaussian kernel for smoothing tile overlaps."""
        # std = size / 3 ensures the gaussian covers the tile nicely without being too peaked
        k = scipy.signal.windows.gaussian(size, std=size / 3.0)
        kernel = np.outer(k, k)
        return kernel

    def _generate_inference_tiles(self, image_id, image_path, h, w):
        """
        Generates tile coordinates with overlap for inference.
        Unlike training, we want overlapping tiles to smooth predictions.
        """
        tiles = []
        stride = int(self.config.TILE_SIZE * (1 - self.config.INFERENCE_OVERLAP))

        # Ensure stride is at least 1
        stride = max(1, stride)

        y_points = list(range(0, h, stride))
        x_points = list(range(0, w, stride))

        # Ensure the last tile covers the edge if not perfectly divisible
        if y_points[-1] + self.config.TILE_SIZE < h:
            y_points.append(max(0, h - self.config.TILE_SIZE))
        if x_points[-1] + self.config.TILE_SIZE < w:
            x_points.append(max(0, w - self.config.TILE_SIZE))

        # Filter redundant points that might occur due to the edge handling above
        y_points = sorted(list(set(y_points)))
        x_points = sorted(list(set(x_points)))

        for y in y_points:
            for x in x_points:
                tile_meta = {
                    "id": image_id,
                    "image_path": image_path,
                    "json_path": "",  # Not needed for test
                    "anatomical_json_path": "",  # Not needed for test
                    "x": x,
                    "y": y,
                    "h": h,
                    "w": w,
                    "has_glom": False,  # Placeholder
                }
                tiles.append(tile_meta)

        return tiles

    def _remove_small_objects(self, mask):
        """Removes connected components smaller than MIN_PIXEL_SIZE."""
        if self.config.MIN_PIXEL_SIZE <= 0:
            return mask

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # stats: [x, y, width, height, area]
        # Label 0 is background

        new_mask = np.zeros_like(mask)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.config.MIN_PIXEL_SIZE:
                new_mask[labels == label] = 1

        return new_mask

    def predict_image(self, image_meta):
        """
        Performs sliding window inference on a single image.
        """
        img_id = image_meta["id"]
        # Construct path relative to input dir (metadata contains relative path)
        # Note: metadata csv usually contains 'test/filename.tiff' in image_path
        # But let's be safe and check how it's stored.
        # Based on metadata script: image_path is 'test/{id}.tiff'

        h_orig = image_meta["height_pixels"]
        w_orig = image_meta["width_pixels"]

        # Limit inference scope in debug mode to avoid timeouts on large images
        h_inf, w_inf = h_orig, w_orig
        if self.config.DEBUG:
            limit = self.config.TILE_SIZE * 4
            h_inf = min(h_orig, limit)
            w_inf = min(w_orig, limit)
            print(f"    Debug: Limiting inference to {w_inf}x{h_inf} region")

        # Generate tiles
        tiles = self._generate_inference_tiles(
            img_id, image_meta["image_path"], h_inf, w_inf
        )

        # Create Dataset and Loader
        # We use 'test' mode transforms (Normalize + ToTensor)
        dataset = HuBMAPDataset(
            tiles, transforms=get_transforms(mode="test"), mode="test"
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize accumulators
        # We use float32 for precision during accumulation
        prob_map = np.zeros((h_orig, w_orig), dtype=np.float32)
        weight_map = np.zeros((h_orig, w_orig), dtype=np.float32)

        gaussian_kernel = self._get_gaussian_kernel(self.config.TILE_SIZE)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).to(self.device).float()

        with torch.no_grad():
            for images, coords in loader:
                images = images.to(self.device, dtype=torch.float)

                # Predict
                outputs = self.model(images)

                # Select Primary Head (Channel 0) and apply Sigmoid
                # Shape: (B, 2, H, W) -> (B, H, W)
                probs = torch.sigmoid(outputs[:, 0, :, :])

                # Process batch
                probs = probs.cpu().numpy()
                coords = coords.numpy()  # (B, 2) -> [x, y]

                # We need the kernel on CPU for numpy operations now
                kernel_cpu = gaussian_kernel.cpu().numpy()

                for i in range(len(images)):
                    prob_tile = probs[i]  # (1024, 1024)
                    x, y = coords[i]

                    # Determine valid region (dataset pads right/bottom)
                    # We need to crop the prediction and the kernel to the actual image area
                    h_read = min(self.config.TILE_SIZE, h_orig - y)
                    w_read = min(self.config.TILE_SIZE, w_orig - x)

                    # Crop
                    valid_prob = prob_tile[:h_read, :w_read]
                    valid_kernel = kernel_cpu[:h_read, :w_read]

                    # Accumulate
                    prob_map[y : y + h_read, x : x + w_read] += (
                        valid_prob * valid_kernel
                    )
                    weight_map[y : y + h_read, x : x + w_read] += valid_kernel

        # Normalize
        # Avoid division by zero
        weight_map[weight_map == 0] = 1.0
        prob_map /= weight_map

        # Threshold
        mask = (prob_map > self.config.MASK_THRESHOLD).astype(np.uint8)

        # Post-processing
        mask = self._remove_small_objects(mask)

        return mask

    def run(self):
        print("Starting Inference...")

        # Load Test Metadata
        test_meta_path = os.path.join(self.config.METADATA_DIR, "test_metadata.csv")
        if not os.path.exists(test_meta_path):
            raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

        test_df = pd.read_csv(test_meta_path)
        print(f"Found {len(test_df)} test images.")

        results = []

        for idx, row in test_df.iterrows():
            img_id = row["id"]
            print(f"Processing image {idx+1}/{len(test_df)}: {img_id}")

            try:
                # Run inference
                mask = self.predict_image(row)

                # Encode
                rle = rle_encode(mask)
                results.append({"id": img_id, "predicted": rle})

            except Exception as e:
                print(f"Error processing {img_id}: {e}")
                # Append empty prediction on error to ensure submission file has all rows
                results.append({"id": img_id, "predicted": ""})

        # Save Submission
        submission_df = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
