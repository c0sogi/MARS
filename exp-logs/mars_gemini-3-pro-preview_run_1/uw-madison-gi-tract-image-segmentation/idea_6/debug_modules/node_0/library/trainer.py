import os
import time
import numpy as np
import torch
import torch.optim as optim
from scipy.ndimage import gaussian_filter

from library.config import Config
from library.model import ResNet3DUNet
from library.losses import BCEDiceLoss
from library.utils import compute_metrics, rle_encode
from library.postprocessing import keep_largest_component_3d


class Trainer:
    """
    Trainer class for 3D Segmentation Model.
    Handles training, validation (with sliding window inference), and prediction.
    """

    def __init__(self, train_loader, val_loader, test_loader=None):
        """
        Args:
            train_loader (DataLoader): DataLoader for training set.
            val_loader (DataLoader): DataLoader for validation set.
            test_loader (DataLoader, optional): DataLoader for test set.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = ResNet3DUNet(
            in_channels=Config.IN_CHANNELS, out_channels=Config.OUT_CHANNELS
        ).to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Cosine Annealing Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Loss Function
        self.criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(self.device)

        # Inference settings
        self.patch_size = Config.PATCH_SIZE  # (D, H, W) -> (32, 224, 224)
        self.overlap = Config.OVERLAP
        self.num_classes = Config.NUM_CLASSES

        # Pre-compute Gaussian weight map for sliding window
        self.weight_map = self._get_gaussian_weight_map(self.patch_size)

    def _get_gaussian_weight_map(self, size, sigma_scale=1.0 / 8.0):
        """
        Generates a Gaussian weight map for patch merging.
        """
        # Create a temporary volume of ones
        tmp = np.zeros(size)
        # Set center pixel to 1
        center = [s // 2 for s in size]
        tmp[tuple(center)] = 1
        # Apply gaussian filter. Sigma is proportional to dimension size.
        # We use a fixed sigma based on the smallest dimension or per dimension.
        # Here we use a heuristic sigma.
        sigmas = [s * sigma_scale for s in size]
        weight_map = gaussian_filter(tmp, sigma=sigmas)

        # Normalize to [0, 1]
        weight_map = weight_map / np.max(weight_map)
        return torch.from_numpy(weight_map).float().to(self.device)

    def _sliding_window_inference(self, volume):
        """
        Performs sliding window inference on a single volume.

        Args:
            volume (torch.Tensor): Input volume of shape (1, D, H, W).

        Returns:
            torch.Tensor: Probability map of shape (C, D, H, W).
        """
        self.model.eval()

        # volume shape: (C_in, D, H, W) where C_in=1
        _, D, H, W = volume.shape

        patch_D, patch_H, patch_W = self.patch_size
        stride_D = int(patch_D * (1 - self.overlap))
        stride_H = int(patch_H * (1 - self.overlap))
        stride_W = int(patch_W * (1 - self.overlap))

        # Pad volume if dimensions are smaller than patch size
        pad_D = max(0, patch_D - D)
        pad_H = max(0, patch_H - H)
        pad_W = max(0, patch_W - W)

        # Also pad to ensure we can cover the edges with strides
        # (Simple approach: pad right/bottom/back)
        volume_padded = torch.nn.functional.pad(
            volume, (0, pad_W, 0, pad_H, 0, pad_D), mode="constant", value=0
        )

        _, D_pad, H_pad, W_pad = volume_padded.shape

        # Output accumulation tensors
        output_sum = torch.zeros(
            (self.num_classes, D_pad, H_pad, W_pad), device=self.device
        )
        count_map = torch.zeros(
            (self.num_classes, D_pad, H_pad, W_pad), device=self.device
        )

        # Expand weight map to match number of classes: (C, D_patch, H_patch, W_patch)
        weight_map_expanded = self.weight_map.unsqueeze(0).repeat(
            self.num_classes, 1, 1, 1
        )

        # Sliding window loop
        # We assume H and W fit the patch size exactly or are handled by padding,
        # but since Config.SPATIAL_SIZE matches patch spatial size, we mostly slide over Depth.
        # However, loop over all dims for robustness.

        z_steps = list(range(0, D_pad - patch_D + 1, stride_D))
        if z_steps[-1] != D_pad - patch_D:
            z_steps.append(D_pad - patch_D)

        y_steps = list(range(0, H_pad - patch_H + 1, stride_H))
        if not y_steps:
            y_steps = [0]  # Handle exact match case
        elif y_steps[-1] != H_pad - patch_H:
            y_steps.append(H_pad - patch_H)

        x_steps = list(range(0, W_pad - patch_W + 1, stride_W))
        if not x_steps:
            x_steps = [0]
        elif x_steps[-1] != W_pad - patch_W:
            x_steps.append(W_pad - patch_W)

        with torch.no_grad():
            for z in z_steps:
                for y in y_steps:
                    for x in x_steps:
                        # Extract patch: (1, 1, D_p, H_p, W_p)
                        patch = volume_padded[
                            :, z : z + patch_D, y : y + patch_H, x : x + patch_W
                        ].unsqueeze(0)
                        patch = patch.to(self.device)

                        # Inference
                        logits = self.model(patch)  # (1, C, D_p, H_p, W_p)
                        probs = torch.sigmoid(logits).squeeze(0)  # (C, D_p, H_p, W_p)

                        # Accumulate
                        output_sum[
                            :, z : z + patch_D, y : y + patch_H, x : x + patch_W
                        ] += (probs * weight_map_expanded)
                        count_map[
                            :, z : z + patch_D, y : y + patch_H, x : x + patch_W
                        ] += weight_map_expanded

        # Average
        # Avoid division by zero
        count_map[count_map == 0] = 1.0
        output_final = output_sum / count_map

        # Crop back to original size
        output_final = output_final[:, :D, :H, :W]

        return output_final

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        count = 0

        start_time = time.time()

        for batch in self.train_loader:
            # batch['image']: (B, 1, D, H, W)
            # batch['mask']: (B, C, D, H, W)
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count
        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {epoch_loss:.6f} - Time: {duration:.2f}s"
        )
        return epoch_loss

    def validate(self, epoch):
        self.model.eval()

        val_dice_scores = []
        val_hausdorff_scores = []
        val_combined_scores = []

        print(f"Starting Validation for Epoch {epoch+1}...")

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # Batch size is 1 for validation
                # image: (1, 1, D, H, W)
                image = batch["image"][0]  # (1, D, H, W)
                mask = batch["mask"][0]  # (C, D, H, W)

                # Sliding Window Inference
                # Returns (C, D, H, W) probabilities
                pred_probs = self._sliding_window_inference(image)

                # Thresholding
                pred_mask = (pred_probs > 0.5).float().cpu().numpy()
                gt_mask = mask.numpy()

                # Metrics per class
                case_dices = []
                case_hausdorffs = []

                for c in range(self.num_classes):
                    p_c = pred_mask[c]
                    g_c = gt_mask[c]

                    # Post-processing: Keep Largest Component
                    p_c_processed = keep_largest_component_3d(p_c)

                    metrics = compute_metrics(p_c_processed, g_c)
                    case_dices.append(metrics["dice"])
                    case_hausdorffs.append(metrics["hausdorff"])

                # Average over classes for this case
                avg_dice = np.mean(case_dices)
                avg_hausdorff = np.mean(case_hausdorffs)

                # Combined score: 0.4*Dice + 0.6*(1-Hausdorff)
                # Note: compute_metrics returns normalized Hausdorff (0-1),
                # where 0 is best. The competition metric maximizes score.
                # Score = 0.4 * Dice + 0.6 * (1 - Hausdorff)
                avg_score = 0.4 * avg_dice + 0.6 * (1.0 - avg_hausdorff)

                val_dice_scores.append(avg_dice)
                val_hausdorff_scores.append(avg_hausdorff)
                val_combined_scores.append(avg_score)

        mean_dice = np.mean(val_dice_scores)
        mean_hausdorff = np.mean(val_hausdorff_scores)
        mean_score = np.mean(val_combined_scores)

        print(
            f"Epoch {epoch+1} Validation - Mean Dice: {mean_dice:.6f}, Mean Hausdorff: {mean_hausdorff:.6f}, Combined Score: {mean_score:.6f}"
        )

        return mean_score

    def fit(self):
        best_score = -float("inf")
        patience = 5
        counter = 0

        print("Starting training...")

        for epoch in range(Config.EPOCHS):
            # Train
            self.train_epoch(epoch)

            # Validate
            score = self.validate(epoch)

            # Scheduler Step
            self.scheduler.step()

            # Checkpoint & Early Stopping
            if score > best_score:
                best_score = score
                counter = 0
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with score: {best_score:.6f}")
            else:
                counter += 1
                print(f"Score did not improve. Counter: {counter}/{patience}")

            if counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Score: {best_score:.6f}")

    def predict_and_submit(self):
        """
        Generates predictions for the test set and creates the submission file.
        """
        print("Loading best model for inference...")
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model weights."
            )

        self.model.eval()
        submission_rows = []

        print("Starting inference on test set...")
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                # batch['image']: (1, 1, D, H, W)
                # batch['id']: list of strings ["case_day"]
                image = batch["image"][0]  # (1, D, H, W)
                case_day_id = batch["id"][0]

                # Sliding Window Inference
                pred_probs = self._sliding_window_inference(image)
                pred_mask = (pred_probs > 0.5).float().cpu().numpy()

                # Process each class
                for c_idx, class_name in enumerate(Config.CLASSES):
                    # Extract 3D volume for class
                    vol_c = pred_mask[c_idx]  # (D, H, W)

                    # Post-processing
                    vol_c_processed = keep_largest_component_3d(vol_c)

                    # Convert back to 2D slices for submission format
                    # Submission requires: id, class, predicted (RLE)
                    # id format: caseXXX_dayYY_slice_ZZZZ

                    D, H, W = vol_c_processed.shape

                    # We need to iterate over slices to generate RLEs
                    # Note: The original slice IDs (0001, 0002...) correspond to indices 0, 1...
                    # assuming the volume was constructed from sorted slices.

                    for d in range(D):
                        slice_mask = vol_c_processed[d, :, :]

                        # Only add entry if mask is not empty (optimization for file size)
                        # However, competition usually requires all rows or specific rows.
                        # The sample submission has rows for every slice/class.
                        # We should generate RLE for every slice.

                        rle_str = rle_encode(slice_mask)

                        # Construct ID
                        # case_day_id is "caseXXX_dayYY"
                        # slice number needs to be 4 digits, 1-based index
                        slice_num_str = f"{d+1:04d}"
                        row_id = f"{case_day_id}_slice_{slice_num_str}"

                        submission_rows.append(f"{row_id},{class_name},{rle_str}")

        # Save submission
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            f.write("id,class,predicted\n")
            f.write("\n".join(submission_rows))

        print(f"Submission file saved to {submission_path}")
