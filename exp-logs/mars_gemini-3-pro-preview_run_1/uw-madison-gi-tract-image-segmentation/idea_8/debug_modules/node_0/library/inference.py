import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    TEST_CSV,
    BATCH_SIZE,
    NUM_WORKERS,
    CLASSES,
    IMG_SIZE,
    THRESHOLD,
)
from library.utils import rle_encode, keep_largest_component, set_seed
from library.dataset import UWMadisonDataset
from library.model import RecurrentUNet


class InferenceEngine:
    def __init__(self, load_cached_data=True):
        """
        Initialize the Inference Engine.

        Args:
            load_cached_data (bool): Whether to use cached preprocessed data.
        """
        set_seed()
        self.device = DEVICE
        self.load_cached_data = load_cached_data

        # Ensure submission directory exists
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # Initialize Model
        self.model = RecurrentUNet(backbone="resnet34", pretrained=False)
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint_name="best_model.pth"):
        """
        Loads model weights from the checkpoint directory.
        """
        path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        print(f"Loading model weights from {path}...")
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_volume(self):
        """
        Performs inference on the test dataset.
        1. Iterates over the test loader.
        2. Aggregates predictions by case_day.
        3. Reconstructs 3D volumes.
        4. Applies 3D CCA post-processing.
        5. Generates RLE submission.
        """
        print("Initializing Test Dataset...")
        dataset = UWMadisonDataset(
            TEST_CSV, mode="test", load_cached_data=self.load_cached_data
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        print("Starting Inference...")

        # Dictionary to store predictions: results[case_day][slice_idx] = (C, H, W) probability map
        # We store raw probabilities or binary masks?
        # To perform CCA efficiently, we need the full volume.
        # Storing binary masks (uint8) saves memory.
        case_buffers = defaultdict(list)

        with torch.no_grad():
            for images, ids in loader:
                # images: (B, T, 1, H, W) -> Model expects (B, 1, T, H, W)
                images = images.permute(0, 2, 1, 3, 4).to(self.device)

                # Forward pass
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # Binarize immediately to save memory (uint8)
                preds = (probs > THRESHOLD).cpu().numpy().astype(np.uint8)

                for b in range(len(ids)):
                    slice_id = ids[b]
                    # Parse ID: caseXXX_dayYY_slice_ZZZZ
                    parts = slice_id.split("_")
                    case_day = f"{parts[0]}_{parts[1]}"
                    slice_idx = int(parts[3])  # Keep as int for sorting

                    case_buffers[case_day].append(
                        {
                            "id": slice_id,
                            "slice_idx": slice_idx,
                            "mask": preds[b],  # (C, H, W)
                        }
                    )

        print("Inference complete. Starting Post-processing (3D CCA) and Encoding...")

        submission_rows = []

        # Process each case volume
        for case_day, slices in case_buffers.items():
            # Sort slices by Z-index to ensure correct 3D reconstruction
            slices.sort(key=lambda x: x["slice_idx"])

            # Stack to create 3D volume: (Depth, C, H, W)
            # Transpose to (C, Depth, H, W) for easier per-class processing
            vol_stack = np.stack([s["mask"] for s in slices], axis=0)  # (D, C, H, W)
            vol_stack = vol_stack.transpose(1, 0, 2, 3)  # (C, D, H, W)

            num_classes, depth, height, width = vol_stack.shape

            # Process each class
            processed_vol = np.zeros_like(vol_stack)

            for c in range(num_classes):
                # Extract 3D binary mask for class c
                class_vol = vol_stack[c]

                # Apply 3D Connected Component Analysis (Keep Largest)
                processed_vol[c] = keep_largest_component(class_vol)

            # Transpose back to (D, C, H, W) to iterate by slice
            processed_vol = processed_vol.transpose(1, 0, 2, 3)

            # Generate RLE for each slice
            for i, slice_data in enumerate(slices):
                current_id = slice_data["id"]

                for c_idx, class_name in enumerate(CLASSES):
                    mask_slice = processed_vol[i, c_idx]

                    # RLE Encode
                    rle = rle_encode(mask_slice)

                    submission_rows.append(
                        {"id": current_id, "class": class_name, "predicted": rle}
                    )

        # Save Submission
        print("Saving submission file...")
        df = pd.DataFrame(submission_rows)
        # Ensure column order
        df = df[["id", "class", "predicted"]]

        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")


def run_inference(load_cached_data=True):
    """
    Helper function to run the full inference pipeline.
    """
    engine = InferenceEngine(load_cached_data=load_cached_data)
    try:
        engine.load_checkpoint("best_model.pth")
    except FileNotFoundError:
        print("Warning: 'best_model.pth' not found. Checking for fallback...")
        # In a real scenario, we might fail here, but for development flow we warn.
        return

    engine.predict_volume()
