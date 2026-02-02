import os
import gc
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import AnatomyAwareUNetPlusPlus
from library.utils import setup_logger, rle_encode
from library.data_processing import get_test_loader


class InferenceRunner:
    """
    Manages the inference pipeline for detecting FTUs in large kidney tissue images.
    Handles model loading, sliding-window prediction with Gaussian blending,
    image reconstruction, and submission file generation.
    """

    def __init__(self, model_path=None):
        """
        Initialize the inference runner.

        Args:
            model_path (str, optional): Path to the trained model weights.
                                        Defaults to Config.MODEL_PATH.
        """
        self.logger = setup_logger()
        self.device = torch.device(Config.DEVICE)
        self.model_path = model_path if model_path else Config.MODEL_PATH

        # Initialize Model
        self.logger.info(f"Initializing model: {Config.MODEL_ARCH}")
        self.model = AnatomyAwareUNetPlusPlus().to(self.device)

        # Load Weights
        self._load_weights()
        self.model.eval()

        # Pre-calculate Gaussian window for tile blending
        self.tile_size = Config.TILE_SIZE
        self.gaussian_window = self._get_gaussian_window(self.tile_size)

    def _load_weights(self):
        """Loads model weights from disk."""
        if os.path.exists(self.model_path):
            self.logger.info(f"Loading weights from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.logger.warning(
                f"Weights not found at {self.model_path}. Using random initialization."
            )

    def _get_gaussian_window(self, size):
        """
        Generates a 2D Gaussian window to weight predictions during reconstruction.
        This reduces edge artifacts when stitching overlapping tiles.
        """
        sigma = size / 2.0
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        x, y = np.meshgrid(x, y)
        d = np.sqrt(x * x + y * y)
        g = np.exp(-(d**2) / (2.0 * 0.5**2))
        return g.astype(np.float32)

    def _process_accumulated_buffers(self, pred_buffer, weight_buffer):
        """
        Normalizes the accumulated weighted predictions and generates the RLE mask.
        """
        # Normalize by the sum of weights (avoid division by zero)
        mask = pred_buffer / (weight_buffer + 1e-6)

        # Threshold to create binary mask
        mask = (mask > Config.MASK_THRESHOLD).astype(np.uint8)

        # Encode
        rle = rle_encode(mask)
        return rle

    def predict_and_submit(self):
        """
        Main inference loop.
        Iterates over test tiles, reconstructs full images, and saves submission CSV.
        """
        self.logger.info("Starting inference pipeline...")
        test_loader = get_test_loader()

        results = []

        # State variables for image reconstruction
        current_id = None
        buffer_pred = None
        buffer_weight = None

        with torch.no_grad():
            for images, coords, ids, shapes in test_loader:
                images = images.to(self.device)

                # Forward pass
                outputs = self.model(images)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs).cpu().numpy()  # Shape: (B, 1, H, W)

                # Convert metadata to numpy for easy indexing
                coords_np = coords.numpy()
                # ids is a tuple of strings from the dataloader

                # Handle shapes tensor (batch of [h, w])
                if isinstance(shapes, torch.Tensor):
                    shapes_np = shapes.numpy()
                else:
                    shapes_np = np.array([s.numpy() for s in shapes])

                # Iterate through the batch
                for i in range(len(images)):
                    img_id = ids[i]
                    y, x = coords_np[i]
                    h, w = shapes_np[i]
                    prob_tile = probs[i, 0]  # Shape: (H_tile, W_tile)

                    # Detect change of image ID
                    if img_id != current_id:
                        # If we have a previous image buffered, process it
                        if current_id is not None:
                            self.logger.info(
                                f"Reconstructing prediction for image: {current_id}"
                            )
                            rle = self._process_accumulated_buffers(
                                buffer_pred, buffer_weight
                            )
                            results.append({"id": current_id, "predicted": rle})

                            # Free memory
                            del buffer_pred, buffer_weight
                            gc.collect()

                        # Initialize buffers for the new image
                        self.logger.info(
                            f"Allocating buffers for image: {img_id} ({h}x{w})"
                        )
                        current_id = img_id
                        buffer_pred = np.zeros((h, w), dtype=np.float32)
                        buffer_weight = np.zeros((h, w), dtype=np.float32)

                    # Accumulate weighted predictions
                    th, tw = prob_tile.shape

                    # Add weighted tile to buffer
                    buffer_pred[y : y + th, x : x + tw] += (
                        prob_tile * self.gaussian_window[:th, :tw]
                    )
                    buffer_weight[y : y + th, x : x + tw] += self.gaussian_window[
                        :th, :tw
                    ]

            # Process the final image in the dataset
            if current_id is not None:
                self.logger.info(f"Reconstructing prediction for image: {current_id}")
                rle = self._process_accumulated_buffers(buffer_pred, buffer_weight)
                results.append({"id": current_id, "predicted": rle})
                del buffer_pred, buffer_weight
                gc.collect()

        # Save Submission
        self.logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df = pd.DataFrame(results)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info("Inference completed successfully.")
