import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import cycle

from library.config import Config
from library.utils import do_kaggle_metric, save_checkpoint, rle_encode
from library.losses import MixedLoss, StudentLoss
from library.models import SpecialistTeacher, GeneralistStudent
from library.data import get_stage1_loaders, get_test_loader, get_stage3_loaders


def log(msg):
    """Helper to print messages immediately."""
    print(msg)
    sys.stdout.flush()


class Pipeline:
    def __init__(self):
        self.device = Config.DEVICE
        self.base_dir = Config.WORKING_DIR
        # We will cache a validation loader for consistent evaluation across stages
        self.fixed_val_loader = None

    def get_val_loader(self):
        """Returns a fixed validation loader (Fold 0) for monitoring."""
        if self.fixed_val_loader is None:
            _, self.fixed_val_loader = get_stage1_loaders(fold=0)
        return self.fixed_val_loader

    def train_supervised_student(self, epochs=Config.EPOCHS):
        """
        Trains the Generalist Student on the labeled dataset (Train split).
        Validates on the Val split with dynamic threshold optimization.
        """
        from library.data import get_supervised_loaders

        log("--- Starting Supervised Student Training ---")

        train_loader, val_loader = get_supervised_loaders()

        model = GeneralistStudent().to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        # Cite solution_lesson_node_00068: Use Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = StudentLoss()

        best_map = 0.0
        best_threshold = 0.5
        patience_counter = 0
        checkpoint_dir = Config.STUDENT_CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                depths = batch["depth"].to(self.device)

                optimizer.zero_grad()
                logits, depth_pred = model(images)

                # Cite solution_lesson_node_00062: Ensure aux head is connected
                loss = criterion(logits, depth_pred, masks, depths)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            scheduler.step()
            avg_loss = train_loss / len(train_loader)

            # Validation with Dynamic Thresholding
            # Cite solution_lesson_node_00033: Align Model Checkpointing via Adaptive Thresholding
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)

                    logits, _ = model(images)
                    probs = torch.sigmoid(logits)

                    val_preds.append(probs.cpu().numpy())
                    val_targets.append(masks.cpu().numpy())

            if len(val_preds) > 0:
                val_preds = np.concatenate(val_preds, axis=0)
                val_targets = np.concatenate(val_targets, axis=0)

                # Search for best threshold for this epoch
                epoch_best_map = 0.0
                epoch_best_t = 0.5
                for t in np.arange(0.3, 0.75, 0.05):
                    score = do_kaggle_metric(val_preds, val_targets, threshold=t)
                    if score > epoch_best_map:
                        epoch_best_map = score
                        epoch_best_t = t
            else:
                epoch_best_map = 0.0
                epoch_best_t = 0.5

            log(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Val mAP: {epoch_best_map:.6f} (Thresh: {epoch_best_t:.2f})"
            )

            if epoch_best_map > best_map:
                best_map = epoch_best_map
                best_threshold = epoch_best_t
                patience_counter = 0
                save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "best_map": best_map,
                        "best_threshold": best_threshold,
                    },
                    is_best=True,
                    checkpoint_dir=checkpoint_dir,
                    filename="student_best.pth",
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                log("Early stopping triggered.")
                break

        return os.path.join(checkpoint_dir, "best_model.pth"), best_threshold

    def optimize_threshold(self, model_path):
        """
        Finds the optimal binarization threshold on the validation set.
        """
        log("--- Optimizing Threshold ---")
        model = GeneralistStudent().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        val_loader = self.get_val_loader()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                # TTA: Average with flip
                logits, _ = model(images)
                probs = torch.sigmoid(logits)

                images_flip = torch.flip(images, dims=[3])
                logits_flip, _ = model(images_flip)
                probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3])

                avg_probs = (probs + probs_flip) / 2.0

                all_preds.append(avg_probs.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        best_t = 0.5
        best_score = 0.0

        # Sweep thresholds
        for t in np.arange(0.3, 0.75, 0.05):
            score = do_kaggle_metric(all_preds, all_targets, threshold=t)
            if score > best_score:
                best_score = score
                best_t = t

        log(f"Best Threshold: {best_t:.2f} (mAP: {best_score:.6f})")
        return best_t

    def generate_submission(self, model_path, threshold):
        """
        Generates final predictions for the test set.
        Handles TTA, cropping, and RLE encoding.
        """
        log("--- Generating Submission ---")
        model = GeneralistStudent().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        test_loader = get_test_loader()

        submission_rows = []

        # Crop parameters to revert padding (128 -> 101)
        # Pad was 13 (top/left) and 14 (bottom/right)
        start = 13
        end = 13 + 101

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                ids = batch["id"]

                # TTA
                logits, _ = model(images)
                probs = torch.sigmoid(logits)

                images_flip = torch.flip(images, dims=[3])
                logits_flip, _ = model(images_flip)
                probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3])

                avg_probs = (probs + probs_flip) / 2.0
                avg_probs = avg_probs.cpu().numpy()  # (B, 1, 128, 128)

                for i, img_id in enumerate(ids):
                    # Extract mask
                    prob_map = avg_probs[i, 0]

                    # Crop center
                    prob_map = prob_map[start:end, start:end]

                    # Binarize
                    binary_mask = (prob_map > threshold).astype(np.uint8)

                    # RLE
                    rle = rle_encode(binary_mask)
                    submission_rows.append([img_id, rle])

        df = pd.DataFrame(submission_rows, columns=["id", "rle_mask"])
        df.to_csv(Config.SUBMISSION_FILE, index=False)
        log(f"Submission saved to {Config.SUBMISSION_FILE}")

    def run(self):
        # Supervised Training with Dynamic Thresholding
        student_path, best_threshold = self.train_supervised_student()

        # Final Output
        self.generate_submission(student_path, best_threshold)
