import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
import cv2

# Import provided library functions and classes
from library.utils import rle_encode, create_tissue_mask, set_seed
from library.model import ConvNeXtUNetPlusPlus


class InferencePipeline:
    def __init__(self, config):
        """
        Initializes the inference pipeline.

        Args:
            config (dict): Configuration dictionary containing:
                - tile_size (int): Input resolution for the model (e.g., 768).
                - stride (int): Stride for sliding window inference.
                - batch_size (int): Batch size for inference.
                - num_classes (int): Number of output classes (1 for binary).
                - model_path (str): Path to the trained model checkpoint.
                - input_dir (str): Root directory of input data.
                - submission_dir (str): Directory to save submission file.
                - working_dir (str): Directory for temporary files/caching.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tile_size = config.get("tile_size", 768)
        self.stride = config.get("stride", 384)
        self.batch_size = config.get("batch_size", 4)

        # Ensure directories exist
        os.makedirs(self.config.get("submission_dir", "./submission"), exist_ok=True)
        os.makedirs(self.config.get("working_dir", "./working/idea_5"), exist_ok=True)

        # Initialize Model
        self.model = ConvNeXtUNetPlusPlus(
            num_classes=config.get("num_classes", 1),
            pretrained=False,  # No need to download weights during inference, loading from checkpoint
        )
        self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Precompute Gaussian Window
        self.gaussian_window = self._get_gaussian_window(self.tile_size)
        self.gaussian_window = (
            torch.from_numpy(self.gaussian_window).to(self.device).float()
        )

    def _load_model(self):
        """Loads the model weights from the specified checkpoint."""
        path = self.config.get("model_path")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at {path}")

        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {path}")

    def _get_gaussian_window(self, size, sigma=-1):
        """
        Generates a 2D Gaussian window for weighting tile predictions.
        """
        if sigma == -1:
            sigma = size / 2.0

        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        x, y = np.meshgrid(x, y)
        d = np.sqrt(x * x + y * y)
        g = np.exp(-(d**2 / (2.0 * (0.5**2))))  # 0.5 is relative sigma

        return g.astype(np.float32)

    def _process_batch(self, images, coords, full_prob_map, weight_sum_map):
        """
        Runs inference on a batch of tiles and accumulates results.
        """
        if len(images) == 0:
            return

        # Normalize and convert to tensor
        # images list of (H, W, C) -> (B, C, H, W)
        batch_tensor = torch.stack(
            [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in images]
        ).to(self.device)

        with torch.no_grad():
            # Model returns list of outputs [fine, medium, coarse...]
            # We use the finest resolution (index 0)
            outputs = self.model(batch_tensor)
            preds = torch.sigmoid(outputs[0])  # (B, 1, H, W)
            preds = preds.squeeze(1)  # (B, H, W)

        # Apply Gaussian weighting and accumulate
        for i, (x, y) in enumerate(coords):
            pred_tile = preds[i] * self.gaussian_window

            # Move to CPU for accumulation into large map
            pred_tile_cpu = pred_tile.cpu().numpy()
            window_cpu = self.gaussian_window.cpu().numpy()

            # Handle boundary conditions (if tile goes out of bounds, though we usually pad input)
            h_tile, w_tile = pred_tile_cpu.shape

            # Accumulate
            full_prob_map[y : y + h_tile, x : x + w_tile] += pred_tile_cpu
            weight_sum_map[y : y + h_tile, x : x + w_tile] += window_cpu

    def predict_single_image(self, image_id, image_path, anatomical_path):
        """
        Performs sliding window inference on a single large image.
        """
        full_path = os.path.join(self.config.get("input_dir", "./input"), image_path)
        anat_full_path = os.path.join(
            self.config.get("input_dir", "./input"), anatomical_path
        )

        # 1. Get Image Dimensions
        with rasterio.open(full_path) as src:
            H, W = src.height, src.width
            # We don't load the full image here to save RAM

        # 2. Generate/Load Tissue Mask
        # This defines the ROI. We only predict within this mask.
        tissue_mask = create_tissue_mask(
            anat_full_path,
            H,
            W,
            load_cached_data=True,
            cache_dir=os.path.join(
                self.config.get("working_dir"), "tissue_masks_cache"
            ),
        )

        # If no tissue is detected, return empty RLE
        if tissue_mask.sum() == 0:
            return ""

        # 3. Initialize Accumulators (CPU)
        # Using float16 might save memory but float32 is safer for accumulation
        prob_map = np.zeros((H, W), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)

        # 4. Define Tile Coordinates
        # We pad the image virtually by handling boundaries in the loop
        x_points = list(range(0, W - self.tile_size + 1, self.stride))
        if (W - self.tile_size) % self.stride != 0:
            x_points.append(W - self.tile_size)  # Ensure last tile covers edge
        if W < self.tile_size:
            x_points = [0]  # Handle small images

        y_points = list(range(0, H - self.tile_size + 1, self.stride))
        if (H - self.tile_size) % self.stride != 0:
            y_points.append(H - self.tile_size)
        if H < self.tile_size:
            y_points = [0]

        # 5. Sliding Window Loop
        batch_imgs = []
        batch_coords = []

        with rasterio.open(full_path) as src:
            for y in y_points:
                for x in x_points:
                    # Check if tile intersects with tissue mask
                    # We can skip background tiles entirely
                    mask_slice = tissue_mask[
                        y : y + self.tile_size, x : x + self.tile_size
                    ]
                    if mask_slice.sum() == 0:
                        continue

                    # Read Image Tile
                    window = Window(x, y, self.tile_size, self.tile_size)
                    img = src.read(window=window)
                    img = np.moveaxis(img, 0, -1)  # (H, W, C)
                    if img.shape[2] > 3:
                        img = img[:, :, :3]  # RGB only

                    # Pad if necessary (e.g. small images)
                    if img.shape[0] != self.tile_size or img.shape[1] != self.tile_size:
                        pad_h = self.tile_size - img.shape[0]
                        pad_w = self.tile_size - img.shape[1]
                        img = np.pad(
                            img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
                        )

                    batch_imgs.append(img)
                    batch_coords.append((x, y))

                    # Process Batch
                    if len(batch_imgs) >= self.batch_size:
                        self._process_batch(
                            batch_imgs, batch_coords, prob_map, weight_map
                        )
                        batch_imgs = []
                        batch_coords = []

            # Process remaining
            if len(batch_imgs) > 0:
                self._process_batch(batch_imgs, batch_coords, prob_map, weight_map)

        # 6. Normalize and Threshold
        # Avoid division by zero
        mask_indices = weight_map > 0
        prob_map[mask_indices] /= weight_map[mask_indices]

        # Apply Tissue Mask constraint (hard filter)
        prob_map = prob_map * tissue_mask

        # Binarize
        binary_mask = (prob_map > 0.5).astype(np.uint8)

        # Clean up memory
        del prob_map, weight_map, tissue_mask
        gc.collect()

        # 7. Encode
        return rle_encode(binary_mask)

    def run(self):
        """
        Executes the inference pipeline on the test set.
        """
        print("Starting Inference Pipeline...")
        set_seed(42)

        # Load Test Metadata
        test_csv_path = "./metadata/test.csv"
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

        test_df = pd.read_csv(test_csv_path)
        results = []

        for idx, row in test_df.iterrows():
            img_id = row["id"]
            img_path = row["image_path"]
            anat_path = row["anatomical_json_path"]

            print(f"Processing {img_id}...")
            try:
                rle = self.predict_single_image(img_id, img_path, anat_path)
                results.append({"id": img_id, "predicted": rle})
            except Exception as e:
                print(f"Error processing {img_id}: {e}")
                results.append({"id": img_id, "predicted": ""})

        # Save Submission
        submission_df = pd.DataFrame(results)
        save_path = os.path.join(self.config.get("submission_dir"), "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
