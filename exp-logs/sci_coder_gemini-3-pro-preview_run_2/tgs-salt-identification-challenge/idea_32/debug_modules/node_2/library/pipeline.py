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

    def train_teacher_fold(self, fold, epochs=Config.STAGE1_EPOCHS):
        """
        Stage 1: Train a single Specialist Teacher fold.
        """
        log(f"--- Starting Stage 1: Specialist Teacher (Fold {fold}) ---")

        train_loader, val_loader = get_stage1_loaders(fold=fold)

        model = SpecialistTeacher().to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = MixedLoss()

        best_map = 0.0
        patience_counter = 0
        best_epoch = 0

        checkpoint_dir = os.path.join(Config.TEACHER_CHECKPOINT_DIR, f"fold_{fold}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                depths = batch["depth"].to(self.device)  # (B, 1) normalized

                optimizer.zero_grad()
                logits = model(images, depths)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = (
                train_loss / len(train_loader) if len(train_loader) > 0 else 0
            )

            # Validation
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)
                    depths = batch["depth"].to(self.device)

                    logits = model(images, depths)
                    probs = torch.sigmoid(logits)

                    val_preds.append(probs.cpu().numpy())
                    val_targets.append(masks.cpu().numpy())

            if len(val_preds) > 0:
                val_preds = np.concatenate(val_preds, axis=0)
                val_targets = np.concatenate(val_targets, axis=0)
                curr_map = float(do_kaggle_metric(val_preds, val_targets))
            else:
                curr_map = 0.0

            log(
                f"Fold {fold} Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.6f} - Val mAP: {curr_map:.6f}"
            )

            # Checkpoint
            is_best = curr_map > best_map
            if is_best:
                best_map = curr_map
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_map": best_map,
                    },
                    is_best=True,
                    checkpoint_dir=checkpoint_dir,
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                log(f"Early stopping triggered at epoch {epoch+1}")
                break

        log(f"Fold {fold} finished. Best mAP: {best_map:.6f} at epoch {best_epoch+1}")
        return best_map, os.path.join(checkpoint_dir, "best_model.pth")

    def run_teacher_ensemble(self):
        """
        Runs training for all folds and filters models based on performance.
        """
        valid_models = []

        for fold in range(Config.STAGE1_FOLDS):
            best_map, model_path = self.train_teacher_fold(fold)

            if best_map >= Config.STAGE1_GATING_THRESHOLD:
                valid_models.append(model_path)
                log(f"Fold {fold} accepted for ensemble.")
            else:
                log(
                    f"Fold {fold} rejected (mAP {best_map:.4f} < {Config.STAGE1_GATING_THRESHOLD})."
                )

        # Fallback: if all failed, take the best one to avoid crash
        if not valid_models:
            log("Warning: No models passed gating. Using Fold 0 as fallback.")
            fallback_path = os.path.join(
                Config.TEACHER_CHECKPOINT_DIR, "fold_0", "best_model.pth"
            )
            if os.path.exists(fallback_path):
                valid_models.append(fallback_path)

        return valid_models

    def generate_marginalized_labels(self, teacher_paths):
        """
        Stage 2: Generate soft pseudo-labels by marginalizing over depth.
        """
        log("--- Starting Stage 2: Marginalized Soft Pseudo-Labeling ---")

        # Load all models
        models = []
        for path in teacher_paths:
            m = SpecialistTeacher().to(self.device)
            ckpt = torch.load(path, map_location=self.device)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            models.append(m)

        test_loader = get_test_loader()
        pseudo_labels = {}  # id -> mask

        # Sigmas to scan (normalized depth values)
        sigmas = Config.DEPTH_SCAN_SIGMAS

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)  # (B, 1, H, W)
                ids = batch["id"]
                B = images.size(0)

                # Accumulator for marginalized probabilities
                batch_probs_sum = torch.zeros_like(images).float()

                # Marginalization: Average over models and depths
                count = 0
                for model in models:
                    for s in sigmas:
                        # Create depth tensor (B, 1) with value s
                        d_tensor = torch.full(
                            (B, 1), s, device=self.device, dtype=torch.float32
                        )

                        logits = model(images, d_tensor)
                        probs = torch.sigmoid(logits)
                        batch_probs_sum += probs
                        count += 1

                avg_probs = batch_probs_sum / count
                avg_probs_np = avg_probs.cpu().numpy()  # (B, 1, 128, 128)

                # Store in dict
                for i, img_id in enumerate(ids):
                    # Remove channel dim: (128, 128)
                    pseudo_labels[img_id] = avg_probs_np[i, 0]

        log(f"Generated pseudo-labels for {len(pseudo_labels)} images.")
        return pseudo_labels

    def train_student_distillation(self, pseudo_labels, epochs=Config.STAGE3_EPOCHS):
        """
        Stage 3: Train Generalist Student on Labeled + Pseudo-Labeled data.
        """
        log("--- Starting Stage 3: Generalist Student Distillation ---")

        labeled_loader, unlabeled_loader = get_stage3_loaders(
            pseudo_labels_dict=pseudo_labels
        )

        # If no unlabeled data (e.g. debugging), handle gracefully
        if unlabeled_loader is None:
            log("Warning: No unlabeled loader returned. Training on labeled only.")
            unlabeled_iter = None
        else:
            unlabeled_iter = cycle(unlabeled_loader)

        model = GeneralistStudent().to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = StudentLoss()

        best_map = 0.0
        patience_counter = 0
        checkpoint_dir = Config.STUDENT_CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)

        val_loader = self.get_val_loader()

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            steps = 0

            for lab_batch in labeled_loader:
                # Labeled Data
                l_img = lab_batch["image"].to(self.device)
                l_mask = lab_batch["mask"].to(self.device)
                l_depth = lab_batch["depth"].to(self.device)  # GT Depth

                # Unlabeled Data
                if unlabeled_iter:
                    try:
                        unlab_batch = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = cycle(unlabeled_loader)
                        unlab_batch = next(unlabeled_iter)

                    u_img = unlab_batch["image"].to(self.device)
                    u_mask = unlab_batch["mask"].to(self.device)  # Soft targets
                    # u_depth is not used/available for loss

                optimizer.zero_grad()

                # Forward Labeled
                l_logits, l_d_pred = model(l_img)
                loss_l = criterion(l_logits, l_d_pred, l_mask, l_depth)

                # Forward Unlabeled
                loss_u = 0.0
                if unlabeled_iter:
                    u_logits, u_d_pred = model(u_img)
                    loss_u = criterion(u_logits, u_d_pred, u_mask, depth_targets=None)

                # Combined Loss
                loss = loss_l + loss_u
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                steps += 1

            scheduler.step()
            avg_loss = train_loss / steps if steps > 0 else 0

            # Validation
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
                curr_map = float(do_kaggle_metric(val_preds, val_targets))
            else:
                curr_map = 0.0

            log(
                f"Student Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Val mAP: {curr_map:.6f}"
            )

            if curr_map > best_map:
                best_map = curr_map
                patience_counter = 0
                save_checkpoint(
                    {"state_dict": model.state_dict(), "best_map": best_map},
                    is_best=True,
                    checkpoint_dir=checkpoint_dir,
                    filename="student_best.pth",
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                log("Student early stopping.")
                break

        return os.path.join(checkpoint_dir, "best_model.pth")

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
        # Stage 1: Train Ensemble
        teacher_paths = self.run_teacher_ensemble()

        # Stage 2: Marginalize
        pseudo_labels = self.generate_marginalized_labels(teacher_paths)

        # Stage 3: Distill
        student_path = self.train_student_distillation(pseudo_labels)

        # Optimization
        best_threshold = self.optimize_threshold(student_path)

        # Final Output
        self.generate_submission(student_path, best_threshold)
