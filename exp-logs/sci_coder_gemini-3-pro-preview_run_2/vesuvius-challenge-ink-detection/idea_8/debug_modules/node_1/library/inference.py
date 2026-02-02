import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library import config
from library import utils
from library import model
from library import dataset


class InferenceRunner:
    def __init__(self, device=config.DEVICE):
        self.device = device
        self.model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
        self.submission_path = config.SUBMISSION_PATH

        # Initialize model architecture
        self.model = model.InkSegFormer(pretrained=False).to(self.device)

    def load_weights(self):
        """Loads the best model weights if available."""
        if not os.path.exists(self.model_path):
            print(f"Error: Model checkpoint not found at {self.model_path}")
            return False

        print(f"Loading model from {self.model_path}...")
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        return True

    def _predict_batch_tta(self, images):
        """
        Predicts on a batch of images using Test Time Augmentation (TTA).
        TTA includes: Original, Horizontal Flip, Vertical Flip.
        """
        # 1. Original
        logits_orig = self.model(images)
        probs_orig = torch.sigmoid(logits_orig)

        # 2. Horizontal Flip
        images_h = torch.flip(images, dims=[3])
        logits_h = self.model(images_h)
        probs_h = torch.flip(torch.sigmoid(logits_h), dims=[3])

        # 3. Vertical Flip
        images_v = torch.flip(images, dims=[2])
        logits_v = self.model(images_v)
        probs_v = torch.flip(torch.sigmoid(logits_v), dims=[2])

        # Average predictions
        avg_probs = (probs_orig + probs_h + probs_v) / 3.0
        return avg_probs

    def run(self):
        # Seed for reproducibility
        utils.np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        # Load weights
        if not self.load_weights():
            print("Skipping inference due to missing model.")
            return

        # Load Test Metadata
        test_csv_path = os.path.join(config.METADATA_DIR, "test.csv")
        if not os.path.exists(test_csv_path):
            print(f"Error: Test metadata not found at {test_csv_path}")
            return

        test_df = pd.read_csv(test_csv_path)
        print(f"Found {len(test_df)} test fragments.")

        # Initialize buffers for full fragment reconstruction
        # We need to know the size of each fragment to reconstruct the mask
        fragment_buffers = {}
        fragment_shapes = {}

        print("Initializing fragment buffers...")
        for _, row in test_df.iterrows():
            frag_id = row["fragment_id"]
            mask_path = os.path.join(config.INPUT_DIR, row["mask_path"])

            # Read mask to get dimensions
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                print(f"Warning: Could not read mask for fragment {frag_id}")
                continue

            h, w = mask_img.shape
            fragment_shapes[frag_id] = (h, w)
            # Use float32 for accumulating probabilities
            fragment_buffers[frag_id] = np.zeros((h, w), dtype=np.float32)

        # Create Dataset and DataLoader
        # transforms=dataset.get_transforms("test") ensures ToTensorV2 is applied
        test_dataset = dataset.InkDataset(
            test_df, mode="test", transforms=dataset.get_transforms("test")
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Starting inference on {len(test_dataset)} tiles...")

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                frag_ids = batch["fragment_id"]
                xs = batch["x"].numpy()
                ys = batch["y"].numpy()

                # Predict with TTA
                probs = self._predict_batch_tta(images)

                # Move to CPU
                probs = probs.cpu().numpy()  # (B, 1, H, W)

                # Place patches into buffers
                for i in range(len(frag_ids)):
                    fid = frag_ids[i]
                    x = xs[i]
                    y = ys[i]
                    prob_patch = probs[i, 0, :, :]  # Remove channel dim

                    if fid not in fragment_buffers:
                        continue

                    # Determine placement coordinates
                    # The patch size is fixed at TILE_SIZE (512)
                    # We simply overwrite. Since the dataset generation uses a sliding window
                    # where the last tile is shifted back to fit, overwriting ensures
                    # the valid data fills the canvas.
                    h_buf, w_buf = fragment_buffers[fid].shape
                    h_patch, w_patch = prob_patch.shape

                    y_end = min(y + h_patch, h_buf)
                    x_end = min(x + w_patch, w_buf)

                    # Crop patch if it goes out of bounds (shouldn't happen with correct logic but safety first)
                    valid_h = y_end - y
                    valid_w = x_end - x

                    fragment_buffers[fid][y:y_end, x:x_end] = prob_patch[
                        :valid_h, :valid_w
                    ]

        # Post-processing and RLE Encoding
        print("Generating submission file...")
        submission_data = []

        for fid in sorted(fragment_buffers.keys()):
            prob_map = fragment_buffers[fid]

            # Apply threshold
            binary_mask = (prob_map > 0.5).astype(np.uint8)

            # Mask out invalid areas using the original mask (optional but good practice)
            # The problem statement implies we output for the whole image, but usually
            # we only care about the valid papyrus area.
            # Let's reload the mask to be safe and mask the prediction.
            # This reduces False Positives in the background.
            mask_path_rel = test_df[test_df["fragment_id"] == fid].iloc[0]["mask_path"]
            mask_path = os.path.join(config.INPUT_DIR, mask_path_rel)
            valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if valid_mask is not None:
                valid_mask = (valid_mask > 0).astype(np.uint8)
                binary_mask = binary_mask * valid_mask

            # RLE Encode
            rle_str = utils.rle_encoding(binary_mask)
            submission_data.append({"Id": fid, "Predicted": rle_str})

        # Save Submission
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")


def run_inference():
    runner = InferenceRunner()
    runner.run()
