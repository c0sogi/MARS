import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import (
    set_seed,
    load_image,
    rle_encode,
    remove_small_objects_3d,
)
from library.models import GhostUNet, EfficientNetUNet


class TestDataset(Dataset):
    """
    Dataset for loading 2.5D stacks for inference.
    Returns the stack, the original image dimensions, and the slice ID.
    """

    def __init__(self, df):
        self.df = df
        # Create a lookup for file paths: (case, day, slice) -> file_path
        self.path_map = {}
        for _, row in self.df.iterrows():
            self.path_map[(row["case"], row["day"], row["slice"])] = row["file_path"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case, day, slice_idx = row["case"], row["day"], row["slice"]

        # Load 2.5D stack
        slices = []
        for offset in [-1, 0, 1]:
            target_slice = slice_idx + offset
            key = (case, day, target_slice)

            # Boundary handling: replicate current slice if neighbor missing
            if key not in self.path_map:
                key = (case, day, slice_idx)

            path = os.path.join(Config.INPUT_DIR, self.path_map[key])
            img = load_image(path)  # (H, W, 1) or (H, W)

            if img.ndim == 3:
                img = img[:, :, 0]

            slices.append(img)

        # Stack -> (H, W, 3)
        stack = np.stack(slices, axis=-1).astype(np.float32)

        # Normalize
        mx = np.max(stack)
        mn = np.min(stack)
        if mx - mn > 0:
            stack = (stack - mn) / (mx - mn)
        else:
            stack = stack - mn

        # Convert to tensor (C, H, W)
        stack_tensor = torch.from_numpy(stack.transpose(2, 0, 1))

        original_h, original_w = stack.shape[:2]

        return {
            "image": stack_tensor,
            "id": row["id"],
            "orig_h": original_h,
            "orig_w": original_w,
            "case": case,
            "day": day,
            "slice": slice_idx,
        }


class InferenceEngine:
    def __init__(self):
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        print("Loading models...")
        self.coarse_model = self._load_model("coarse")
        self.fine_model = self._load_model("fine")

    def _load_model(self, stage):
        if stage == "coarse":
            model = GhostUNet(
                in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES
            )
            path = Config.COARSE_MODEL_PATH
        else:
            model = EfficientNetUNet(
                in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES
            )
            path = Config.FINE_MODEL_PATH

        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded {stage} model from {path}")
        else:
            print(
                f"Warning: Checkpoint for {stage} not found at {path}. Using random weights."
            )

        model.to(self.device)
        model.eval()
        return model

    def extract_roi(self, coarse_mask, orig_h, orig_w):
        """
        Calculates ROI from coarse mask (union of classes).
        coarse_mask: (C, H, W) numpy array, binary.
        """
        # Union of all classes
        mask_union = np.max(coarse_mask, axis=0)  # (H, W)

        rows, cols = np.where(mask_union > 0)

        if len(rows) == 0:
            return None

        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)

        # Scale to original dimensions
        # Coarse mask is 256x256
        scale_y = orig_h / Config.COARSE_IMG_SIZE[0]
        scale_x = orig_w / Config.COARSE_IMG_SIZE[1]

        y_min = int(y_min * scale_y)
        y_max = int(y_max * scale_y)
        x_min = int(x_min * scale_x)
        x_max = int(x_max * scale_x)

        # Add Margin
        box_h = y_max - y_min
        box_w = x_max - x_min
        margin = max(box_h, box_w) * Config.ROI_MARGIN_RATIO

        y_min = max(0, int(y_min - margin))
        y_max = min(orig_h, int(y_max + margin))
        x_min = max(0, int(x_min - margin))
        x_max = min(orig_w, int(x_max + margin))

        return (y_min, y_max, x_min, x_max)

    def predict_case(self, case_df):
        """
        Runs inference for a single case (all slices).
        Returns a dictionary {slice_id: {class: rle}}
        """
        # Sort by slice index to ensure correct 3D ordering
        case_df = case_df.sort_values("slice").reset_index(drop=True)

        dataset = TestDataset(case_df)
        # Process slice by slice (batch size 1 for simplicity with varying crop sizes)
        # We could batch coarse pass, but fine pass needs dynamic crops.
        # Given the constraints, simple loop is robust.

        slice_predictions = []  # List of (C, H, W) arrays
        slice_ids = []

        with torch.no_grad():
            for i in range(len(dataset)):
                data = dataset[i]
                img_tensor = data["image"].to(self.device)  # (3, H, W)
                orig_h, orig_w = data["orig_h"], data["orig_w"]
                slice_ids.append(data["id"])

                # --- Stage 1: Coarse ---
                # Resize to Coarse Size
                img_coarse = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=Config.COARSE_IMG_SIZE,
                    mode="bilinear",
                    align_corners=False,
                )

                coarse_logits = self.coarse_model(img_coarse)
                coarse_probs = torch.sigmoid(coarse_logits)
                coarse_mask = (
                    (coarse_probs > Config.MASK_THRESHOLD).float().cpu().numpy()[0]
                )  # (C, 256, 256)

                # --- ROI Extraction ---
                roi = self.extract_roi(coarse_mask, orig_h, orig_w)

                final_mask = np.zeros(
                    (Config.NUM_CLASSES, orig_h, orig_w), dtype=np.uint8
                )

                # --- Stage 2: Fine ---
                if roi is not None:
                    y1, y2, x1, x2 = roi

                    # Crop original image
                    # img_tensor is (3, H, W)
                    img_crop = img_tensor[:, y1:y2, x1:x2].unsqueeze(
                        0
                    )  # (1, 3, h_crop, w_crop)

                    # Resize to Fine Size
                    img_fine = F.interpolate(
                        img_crop,
                        size=Config.FINE_IMG_SIZE,
                        mode="bilinear",
                        align_corners=False,
                    )

                    fine_logits = self.fine_model(img_fine)
                    fine_probs = torch.sigmoid(fine_logits)
                    fine_pred = (
                        fine_probs > Config.MASK_THRESHOLD
                    ).float()  # (1, C, 320, 320)

                    # Resize back to crop size
                    fine_pred_crop = (
                        F.interpolate(
                            fine_pred, size=(y2 - y1, x2 - x1), mode="nearest"
                        )
                        .cpu()
                        .numpy()[0]
                    )  # (C, h_crop, w_crop)

                    # Paste
                    final_mask[:, y1:y2, x1:x2] = fine_pred_crop.astype(np.uint8)

                slice_predictions.append(final_mask)

        # --- 3D Post-Processing ---
        # Stack: (Depth, C, H, W)
        volume = np.stack(slice_predictions, axis=0)

        # Process each class channel independently
        for c in range(Config.NUM_CLASSES):
            # Extract channel: (Depth, H, W)
            class_vol = volume[:, c, :, :]
            cleaned_vol = remove_small_objects_3d(
                class_vol, min_size=Config.MIN_PIXEL_COUNT
            )
            volume[:, c, :, :] = cleaned_vol

        # --- Encode RLE ---
        results = {}
        for i, sid in enumerate(slice_ids):
            results[sid] = {}
            for c_idx, class_name in enumerate(Config.CLASS_LABELS):
                mask_slice = volume[i, c_idx, :, :]
                rle = rle_encode(mask_slice)
                results[sid][class_name] = rle

        return results

    def generate_submission(self):
        print("Generating submission...")

        # Load Test Metadata
        test_df = pd.read_csv(Config.TEST_META_PATH)

        # We need to iterate by case/day to form volumes
        # Create a grouping key
        test_df["group_key"] = (
            test_df["case"].astype(str) + "_" + test_df["day"].astype(str)
        )

        # Get unique slices (test_df has 3 rows per slice, one for each class)
        # We process unique slices, then map back
        unique_slices_df = (
            test_df[["id", "case", "day", "slice", "file_path", "group_key"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        groups = unique_slices_df.groupby("group_key")

        all_results = {}

        for group_name, group_df in groups:
            # Predict for the whole case/day volume
            case_results = self.predict_case(group_df)
            all_results.update(case_results)

        # Create submission DataFrame
        # The sample submission format is: id, class, predicted
        # test_df already has these columns (predicted is empty/placeholder)

        submission_rows = []

        # Iterate through the original test_df to maintain order and structure
        for idx, row in test_df.iterrows():
            sid = row["id"]
            cls = row["class"]

            rle = ""
            if sid in all_results and cls in all_results[sid]:
                rle = all_results[sid][cls]

            submission_rows.append({"id": sid, "class": cls, "predicted": rle})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_inference():
    engine = InferenceEngine()
    engine.generate_submission()
