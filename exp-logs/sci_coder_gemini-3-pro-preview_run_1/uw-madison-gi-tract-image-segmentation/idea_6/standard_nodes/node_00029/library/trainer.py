import os
import time
import numpy as np
import torch
import torch.optim as optim
from scipy.ndimage import gaussian_filter

from library.config import Config
from library.model import ResNetUNet2D
from library.losses import BCEDiceLoss
from library.utils import compute_metrics, rle_encode
from library.postprocessing import keep_largest_component_3d


class Trainer:
    """
    Trainer class for 2.5D Segmentation Model.
    """

    def __init__(self, train_loader, val_loader, test_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(Config.DEVICE)

        # Initialize Model (2D)
        self.model = ResNetUNet2D(
            in_channels=Config.IN_CHANNELS, out_channels=Config.OUT_CHANNELS
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        self.criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(self.device)
        self.num_classes = Config.NUM_CLASSES

    def _predict_volume(self, volume):
        """
        Performs 2.5D inference on a full volume by iterating slices.
        Args:
            volume (torch.Tensor): (1, D, H, W) input volume.
        Returns:
            torch.Tensor: (C, D, H, W) probabilities.
        """
        self.model.eval()

        # volume: (1, D, H, W) -> squeeze -> (D, H, W)
        vol = volume.squeeze(0)
        D, H, W = vol.shape

        # Prepare output
        probs_vol = torch.zeros((self.num_classes, D, H, W), dtype=torch.float32)

        # We process in batches for speed
        batch_size = 32

        with torch.no_grad():
            for i in range(0, D, batch_size):
                end = min(i + batch_size, D)
                indices = range(i, end)

                batch_imgs = []
                for z in indices:
                    # 2.5D Stack: z-1, z, z+1
                    z_prev = max(0, z - 1)
                    z_next = min(D - 1, z + 1)

                    # Stack slices
                    stack = torch.stack(
                        [vol[z_prev], vol[z], vol[z_next]], dim=0
                    )  # (3, H, W)
                    batch_imgs.append(stack)

                # Create batch tensor
                batch_tensor = torch.stack(batch_imgs).to(self.device)  # (B, 3, H, W)

                # Predict
                logits = self.model(batch_tensor)  # (B, C, H, W)
                probs = torch.sigmoid(logits).cpu()

                # Store
                probs_vol[:, i:end, :, :] = probs.permute(1, 0, 2, 3)

        return probs_vol

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        count = 0

        start_time = time.time()

        for batch in self.train_loader:
            # batch['image']: (B, 3, H, W)
            # batch['mask']: (B, C, H, W)
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
                pred_probs = self._predict_volume(image)

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

                # 2.5D Slice Inference
                pred_probs = self._predict_volume(image)
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
