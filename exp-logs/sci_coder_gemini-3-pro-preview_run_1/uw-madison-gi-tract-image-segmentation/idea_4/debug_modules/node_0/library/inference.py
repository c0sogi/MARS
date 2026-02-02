import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.ndimage
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import UWDataset, get_transforms
from library.model import UnetPlusPlus
from library.utils import rle_encode, set_seed, group_metadata_by_case


class InferencePipeline:
    """
    Manages the inference process: loading model, predicting on test set,
    applying 3D post-processing, and generating submission file.
    """

    def __init__(self, model_path=None):
        self.device = Config.DEVICE
        # Default to best_model.pth in checkpoints dir if not provided
        self.model_path = (
            model_path
            if model_path
            else os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        )
        self.classes = Config.CLASSES
        self.num_classes = Config.NUM_CLASSES

    def load_model(self):
        """
        Loads the trained U-Net++ model.
        Initializes with deep_supervision=True to match trained weights,
        but eval() mode ensures single output during inference.
        """
        # Initialize model structure matching training configuration
        model = UnetPlusPlus(
            classes=self.num_classes, deep_supervision=Config.DEEP_SUPERVISION
        )

        # Load weights
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Model path {self.model_path} does not exist.")

        model.to(self.device)
        model.eval()
        return model

    def post_process_volume(self, volume_mask):
        """
        Applies 3D Connected Component Analysis to keep only the largest object.

        Args:
            volume_mask (np.ndarray): Binary mask of shape (D, H, W).

        Returns:
            np.ndarray: Cleaned binary mask of shape (D, H, W).
        """
        # Label connected components
        labeled_vol, num_features = scipy.ndimage.label(volume_mask)

        # If no features or only 1, return as is
        if num_features <= 1:
            return volume_mask

        # Calculate size of each component
        # label 0 is background, so we look at 1..num_features
        component_sizes = scipy.ndimage.sum(
            volume_mask, labeled_vol, range(1, num_features + 1)
        )

        # Find label of largest component
        largest_label = np.argmax(component_sizes) + 1

        # Create mask for largest component
        cleaned_vol = (labeled_vol == largest_label).astype(np.uint8)

        return cleaned_vol

    def generate_submission(self):
        """
        Runs the full inference pipeline and saves submission.csv.
        """
        set_seed(Config.SEED)

        # 1. Load Test Metadata
        if not os.path.exists(Config.TEST_CSV):
            print("Test metadata not found. Cannot run inference.")
            return

        df_test = pd.read_csv(Config.TEST_CSV)

        # 2. Group by Case/Day for 3D Consistency
        grouped_cases = group_metadata_by_case(df_test)

        # 3. Load Model
        model = self.load_model()

        # 4. Setup Transforms (Resize only)
        transforms = get_transforms(mode="test")

        submission_data = []

        print(f"Starting inference on {len(grouped_cases)} cases...")

        # Iterate over each case (patient/day)
        # Sorting keys ensures deterministic order
        for case_day_key in sorted(grouped_cases.keys()):
            case_df = grouped_cases[case_day_key]

            # Get original dimensions for this case (assume consistent within case)
            if len(case_df) > 0:
                orig_h = case_df.iloc[0]["height"]
                orig_w = case_df.iloc[0]["width"]
            else:
                continue

            # Create Dataset/Loader for this case
            ds = UWDataset(case_df, mode="test", transforms=transforms)
            loader = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY,
            )

            # Collect predictions for the whole volume
            # We will store raw probabilities: List of tensors (B, C, H, W)
            volume_probs_list = []

            with torch.no_grad():
                for batch in loader:
                    imgs = batch["image"].to(self.device)

                    # Forward pass
                    # model.eval() is set, so returns single tensor (B, C, 320, 320)
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)

                    # Resize back to original resolution immediately to save memory
                    # torch.nn.functional.interpolate expects (N, C, H, W)
                    probs_resized = F.interpolate(
                        probs,
                        size=(orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                    volume_probs_list.append(probs_resized.cpu())

            # Concatenate to form full volume tensor: (Depth, C, H, W)
            if not volume_probs_list:
                continue

            volume_tensor = torch.cat(volume_probs_list, dim=0)

            # Threshold to binary
            volume_mask_all = (volume_tensor > 0.5).numpy().astype(np.uint8)

            # Process each class channel independently
            # volume_mask_all shape: (Depth, Classes, Height, Width)

            for cls_idx, cls_name in enumerate(self.classes):
                # Extract 3D volume for this class: (Depth, H, W)
                class_vol = volume_mask_all[:, cls_idx, :, :]

                # Apply 3D Post-Processing (CCA)
                processed_vol = self.post_process_volume(class_vol)

                # Encode each slice and add to submission
                for i in range(len(case_df)):
                    slice_id = case_df.iloc[i]["id"]
                    mask_slice = processed_vol[i, :, :]

                    rle_str = rle_encode(mask_slice)

                    submission_data.append([slice_id, cls_name, rle_str])

        # 5. Save Submission
        sub_df = pd.DataFrame(submission_data, columns=["id", "class", "predicted"])

        # Ensure output directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

        sub_df.to_csv(save_path, index=False)
        print(f"Inference complete. Submission saved to {save_path}")
