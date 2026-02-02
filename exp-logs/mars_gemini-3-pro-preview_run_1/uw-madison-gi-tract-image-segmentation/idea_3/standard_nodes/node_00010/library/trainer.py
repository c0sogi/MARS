import os
import gc
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict

from library.config import Config
from library.utils import (
    set_seed,
    rle_decode,
    rle_encode,
    compute_dice_coefficient,
    compute_hausdorff_distance,
    keep_largest_component,
)
from library.model import UNet25D
from library.loss import BCEDiceLoss


class Trainer:
    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = Config.DEVICE

        # Model Setup
        self.model = UNet25D(backbone_name=Config.BACKBONE, pretrained=True)
        self.model.to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )
        self.criterion = BCEDiceLoss(
            bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT
        )
        self.scaler = GradScaler()

        # Training State
        self.best_score = -float("inf")
        self.history = []

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {epoch_loss:.6f} | LR: {current_lr:.2e}"
        )
        return epoch_loss

    def validate(self):
        """
        Performs validation by reconstructing 3D volumes from 2D predictions.
        Applies 3D post-processing (Connected Component Analysis) before scoring.
        """
        self.model.eval()

        # Store predictions: preds[case_day] = list of (slice_int, mask_numpy)
        preds_map = defaultdict(list)

        # 1. Inference Loop
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                # slice_info format: {case}_{day}_{slice}
                slice_infos = batch["slice_info"]

                with autocast():
                    outputs = self.model(images)
                    probs = torch.sigmoid(outputs)

                # Convert to binary mask on CPU
                pred_masks = (probs > Config.MASK_THRESHOLD).float().cpu().numpy()

                for i, slice_info in enumerate(slice_infos):
                    parts = slice_info.split("_")
                    case_day = f"{parts[0]}_{parts[1]}"
                    slice_num = int(parts[2])
                    preds_map[case_day].append((slice_num, pred_masks[i]))

        # 2. 3D Reconstruction and Scoring
        # Load validation metadata to construct Ground Truth
        df_val = pd.read_csv(Config.VAL_CSV, keep_default_na=False)

        # Group dataframe by case_day for efficient GT construction
        val_groups = df_val.groupby(["case", "day"])

        dice_scores = []
        hausdorff_scores = []

        for (case, day), group in val_groups:
            case_day = f"{case}_{day}"
            if case_day not in preds_map:
                continue

            # Sort predictions by slice number
            case_preds = sorted(preds_map[case_day], key=lambda x: x[0])

            # Determine volume dimensions
            # Assuming all slices in a case have same H, W
            h = group.iloc[0]["height"]
            w = group.iloc[0]["width"]
            num_slices = len(group)

            # Construct Volumes (D, H, W, C)
            gt_vol = np.zeros((num_slices, h, w, Config.NUM_CLASSES), dtype=np.uint8)
            pred_vol = np.zeros((num_slices, h, w, Config.NUM_CLASSES), dtype=np.uint8)

            # Map slice number to index in volume (0 to N-1)
            # Slices in group are not guaranteed to be contiguous integers, but usually are.
            # We map based on sorted order in group.
            sorted_group = group.sort_values(
                "slice", key=lambda x: x.astype(int) if x.dtype == "O" else x
            )
            slice_to_idx = {
                int(row.slice): i for i, row in enumerate(sorted_group.itertuples())
            }

            # Fill GT Volume
            for row in sorted_group.itertuples():
                idx = slice_to_idx[int(row.slice)]
                for c_idx, class_name in enumerate(Config.CLASSES):
                    rle = getattr(row, class_name)
                    if rle:
                        gt_vol[idx, :, :, c_idx] = rle_decode(rle, (h, w))

            # Fill Pred Volume
            for slice_num, mask in case_preds:
                if slice_num in slice_to_idx:
                    idx = slice_to_idx[slice_num]
                    # Resize mask if needed (model output is Config.IMG_SIZE)
                    if mask.shape[1:] != (h, w):
                        # mask is (C, H_model, W_model) -> transpose to (H, W, C) for resize -> back
                        mask_t = mask.transpose(1, 2, 0)
                        mask_resized = cv2.resize(
                            mask_t, (w, h), interpolation=cv2.INTER_NEAREST
                        )
                        if mask_resized.ndim == 2:
                            mask_resized = np.expand_dims(mask_resized, axis=-1)
                        pred_vol[idx] = mask_resized
                    else:
                        pred_vol[idx] = mask.transpose(1, 2, 0)

            # 3. Post-Processing: 3D Connected Components
            # Process each class channel independently
            for c in range(Config.NUM_CLASSES):
                pred_vol[..., c] = keep_largest_component(pred_vol[..., c])

            # 4. Compute Metrics
            # Dice (per class, then mean)
            d_score = compute_dice_coefficient(gt_vol, pred_vol)
            dice_scores.append(d_score)

            # Hausdorff (per class, then mean)
            # We compute HD for each class and average
            hd_c_scores = []
            for c in range(Config.NUM_CLASSES):
                hd_c = compute_hausdorff_distance(gt_vol[..., c], pred_vol[..., c])
                hd_c_scores.append(hd_c)
            hausdorff_scores.append(np.mean(hd_c_scores))

        # Aggregate Metrics
        mean_dice = np.mean(dice_scores)
        mean_hausdorff = np.mean(hausdorff_scores)

        # Competition Score: 0.4 * Dice + 0.6 * (1 - Hausdorff)
        # Note: Hausdorff is distance (0 is best), so we invert it for scoring.
        # The prompt implies 0-1 bounded score.
        score = 0.4 * mean_dice + 0.6 * (1.0 - mean_hausdorff)

        print(
            f"Validation | Dice: {mean_dice:.6f} | Hausdorff: {mean_hausdorff:.6f} | Score: {score:.6f}"
        )
        return score

    def fit(self):
        print("Starting training...")
        patience = 5
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_score = self.validate()
            self.scheduler.step()

            duration = time.time() - start_time
            print(f"Epoch completed in {duration:.0f}s")

            # Save Best Model
            if val_score > self.best_score:
                print(
                    f"Score improved from {self.best_score:.6f} to {val_score:.6f}. Saving model..."
                )
                self.best_score = val_score
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

            print("-" * 30)

        # Generate Submission after training
        self.predict_submission()

    def predict_submission(self):
        """
        Generates predictions for the test set, applies 3D post-processing,
        and saves the submission.csv file.
        """
        print("Generating submission...")

        # Load Best Model
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model weights.")
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()
        preds_map = defaultdict(list)

        # 1. Inference
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["image"].to(self.device)
                slice_infos = batch["slice_info"]

                with autocast():
                    outputs = self.model(images)
                    probs = torch.sigmoid(outputs)

                pred_masks = (probs > Config.MASK_THRESHOLD).float().cpu().numpy()

                for i, slice_info in enumerate(slice_infos):
                    parts = slice_info.split("_")
                    case_day = f"{parts[0]}_{parts[1]}"
                    slice_num = int(parts[2])
                    preds_map[case_day].append((slice_num, pred_masks[i]))

        # 2. Process Volumes & Format Submission
        df_test = pd.read_csv(Config.TEST_CSV, keep_default_na=False)
        # We need original dimensions to resize back
        # Create a lookup for dimensions
        dims_lookup = {}
        for row in df_test.itertuples():
            dims_lookup[row.id] = (row.height, row.width)

        submission_rows = []

        # Iterate over each case_day group in predictions
        for case_day, slices_data in preds_map.items():
            # Sort by slice number
            slices_data.sort(key=lambda x: x[0])

            # Get dimensions from the first slice of this case (assuming constant)
            # Construct an ID to query lookup: case_day_slice_XXXX
            sample_slice_num = slices_data[0][0]
            sample_id = f"{case_day}_slice_{sample_slice_num:04d}"

            # Fallback if specific ID not found (should not happen if logic is correct)
            if sample_id in dims_lookup:
                h, w = dims_lookup[sample_id]
            else:
                # Fallback to model output size if metadata lookup fails
                h, w = Config.IMG_SIZE, Config.IMG_SIZE

            # Determine volume depth
            # Note: slices_data might not be contiguous if test set is partial,
            # but usually test set contains full scans.
            # We will construct a volume based on the number of slices we have.
            num_slices = len(slices_data)

            # Create Volume (D, H, W, C)
            # We use model output size for CCA to save memory/compute, then resize 2D slices later
            # Or resize first? Resizing first is better for accurate CCA on full res.
            # But full res 3D is heavy.
            # Strategy: Perform CCA on model output resolution (320x320), then resize to original.
            vol_h, vol_w = Config.IMG_SIZE, Config.IMG_SIZE
            volume = np.zeros(
                (num_slices, vol_h, vol_w, Config.NUM_CLASSES), dtype=np.uint8
            )

            for idx, (_, mask) in enumerate(slices_data):
                # mask is (C, H, W) -> transpose to (H, W, C)
                volume[idx] = mask.transpose(1, 2, 0)

            # Apply 3D CCA
            for c in range(Config.NUM_CLASSES):
                volume[..., c] = keep_largest_component(volume[..., c])

            # Generate RLEs
            for idx, (slice_num, _) in enumerate(slices_data):
                # ID format: case123_day20_slice_0001
                current_id = f"{case_day}_slice_{slice_num:04d}"

                # Get original dimensions for this specific slice
                if current_id in dims_lookup:
                    orig_h, orig_w = dims_lookup[current_id]
                else:
                    orig_h, orig_w = h, w

                for c_idx, class_name in enumerate(Config.CLASSES):
                    # Extract mask for this slice and class
                    mask_slice = volume[idx, :, :, c_idx]

                    # Resize to original resolution
                    if (vol_h, vol_w) != (orig_h, orig_w):
                        mask_slice = cv2.resize(
                            mask_slice,
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    rle = rle_encode(mask_slice)

                    submission_rows.append(
                        {"id": current_id, "class": class_name, "predicted": rle}
                    )

        # Create DataFrame
        df_sub = pd.DataFrame(submission_rows)

        # Ensure submission matches sample_submission order and structure
        # Cite debug_lesson_4: Align Submission Order to Match Reconstruction Logic
        sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
        if os.path.exists(sample_sub_path):
            sample_df = pd.read_csv(sample_sub_path)

            # Create a lookup dictionary for fast mapping
            # Key: (id, class), Value: predicted_rle
            pred_lookup = dict(
                zip(zip(df_sub["id"], df_sub["class"]), df_sub["predicted"])
            )

            # Map predictions to sample_submission order
            # Use empty string for missing predictions
            keys = zip(sample_df["id"], sample_df["class"])
            sample_df["predicted"] = [pred_lookup.get(k, "") for k in keys]

            df_sub = sample_df

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
