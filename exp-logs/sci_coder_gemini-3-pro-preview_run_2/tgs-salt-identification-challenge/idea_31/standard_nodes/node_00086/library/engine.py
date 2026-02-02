import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import get_batch_iou_score
from library.losses import TeacherLoss, StudentLoss


class Engine:
    """
    Core engine for training, validation, and inference.
    Implements the logic for:
    1. Specialist Teacher Training (Depth-Injected)
    2. Marginalized Depth Scanning (Pseudo-label generation)
    3. Generalist Student Training (Multi-Task Distillation)
    """

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def save_checkpoint(model, path):
        """Saves the model state dictionary."""
        torch.save(model.state_dict(), path)

    @staticmethod
    def train_teacher_epoch(model, loader, optimizer, device, loss_fn, scheduler=None):
        """
        Stage 1: Trains the Specialist Teacher with Depth Injection.
        """
        model.train()
        running_loss = 0.0

        for batch in loader:
            images, masks, depths, _ = batch

            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            # Depth Jitter: Add Gaussian noise to depth to prevent overfitting
            if Config.TEACHER_DEPTH_JITTER_STD > 0:
                noise = torch.randn_like(depths) * Config.TEACHER_DEPTH_JITTER_STD
                depths = depths + noise

            optimizer.zero_grad()

            # Teacher Forward: Requires image and depth
            outputs = model(images, depths)

            loss = loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        return running_loss / len(loader)

    @staticmethod
    def train_student_epoch(
        model,
        labeled_loader,
        unlabeled_loader,
        optimizer,
        device,
        loss_fn,
        scheduler=None,
    ):
        """
        Stage 3: Trains the Generalist Student using Multi-Task Distillation.
        Iterates over both labeled (Train) and unlabeled (Test w/ Pseudo-labels) data.
        """
        model.train()
        running_loss = 0.0

        # Cycle through unlabeled loader if it's smaller (or vice versa),
        # but here we simply assume we iterate based on the labeled loader length
        # and cycle the unlabeled one.
        unlabeled_iter = iter(unlabeled_loader)

        for labeled_batch in labeled_loader:
            # Fetch unlabeled batch
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            # --- Prepare Data ---
            # Labeled Data
            l_images, l_masks, l_depths, _ = labeled_batch
            l_images = l_images.to(device)
            l_masks = l_masks.to(device)
            l_depths = l_depths.to(device)

            # Unlabeled Data (Pseudo-labels)
            # Note: u_masks here are the Soft Pseudo-Labels generated in Stage 2
            u_images, u_masks, _, _ = unlabeled_batch
            u_images = u_images.to(device)
            u_masks = u_masks.to(device)

            optimizer.zero_grad()

            # --- Forward & Loss ---

            # 1. Supervised Step (Labeled)
            # Student returns (mask_logits, depth_pred)
            l_pred_mask, l_pred_depth = model(l_images)
            loss_sup = loss_fn(
                l_pred_mask, l_pred_depth, l_masks, target_depth=l_depths
            )

            # 2. Distillation Step (Unlabeled)
            u_pred_mask, u_pred_depth = model(u_images)
            # Loss vs Soft Targets (BCE only), ignore depth regression for unlabeled
            loss_unsup = loss_fn(u_pred_mask, u_pred_depth, u_masks, target_depth=None)

            # Combine losses
            loss = loss_sup + loss_unsup

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        return running_loss / len(labeled_loader)

    @staticmethod
    def validate(model, loader, device, loss_fn, mode="teacher"):
        """
        Evaluates the model on the validation set.
        Returns average loss and mAP score.
        """
        model.eval()
        running_loss = 0.0
        running_score = 0.0

        with torch.no_grad():
            for batch in loader:
                images, masks, depths, _ = batch

                images = images.to(device)
                masks = masks.to(device)
                depths = depths.to(device)

                if mode == "teacher":
                    outputs = model(images, depths)
                    loss = loss_fn(outputs, masks)
                    preds = torch.sigmoid(outputs)
                else:
                    # Student mode
                    outputs, pred_depth = model(images)
                    # Validation loss includes depth regression component for monitoring
                    loss = loss_fn(outputs, pred_depth, masks, target_depth=depths)
                    preds = torch.sigmoid(outputs)

                running_loss += loss.item()

                # Calculate mAP (IoU)
                running_score += get_batch_iou_score(preds, masks)

        return running_loss / len(loader), running_score / len(loader)

    @staticmethod
    def predict_depth_scan(model, loader, device):
        """
        Stage 2: Marginalized Depth-Scan.
        Generates soft pseudo-labels for the test set by averaging predictions
        across a range of plausible depth values.
        """
        model.eval()
        results = {}  # id -> soft_mask (numpy)

        # Define scan range in standard deviations
        min_z = Config.DEPTH_SCAN_MIN_STD
        max_z = Config.DEPTH_SCAN_MAX_STD
        steps = Config.DEPTH_SCAN_STEPS
        z_values = np.linspace(min_z, max_z, steps)

        with torch.no_grad():
            for batch in loader:
                images, _, ids = batch  # Test loader: img, dummy_depth, id
                images = images.to(device)

                b_size = images.size(0)
                h, w = images.size(2), images.size(3)

                accumulated_probs = torch.zeros((b_size, 1, h, w), device=device)

                for z_val in z_values:
                    # Create constant depth tensor for this scan step
                    # Since the network expects normalized depth, we pass the z-score directly
                    d_tensor = torch.full(
                        (b_size, 1), z_val, device=device, dtype=torch.float32
                    )

                    logits = model(images, d_tensor)
                    probs = torch.sigmoid(logits)
                    accumulated_probs += probs

                # Marginalize (Average)
                avg_probs = accumulated_probs / len(z_values)
                avg_probs = avg_probs.cpu().numpy()

                # Store results
                for i, img_id in enumerate(ids):
                    results[img_id] = avg_probs[i, 0]  # (H, W)

        return results

    @staticmethod
    def predict_tta(model, loader, device):
        """
        Inference with Test-Time Augmentation (Horizontal Flip).
        Used for the final submission with the Student model.
        """
        model.eval()
        results = {}

        with torch.no_grad():
            for batch in loader:
                images, _, ids = batch
                images = images.to(device)

                # 1. Original
                out_orig, _ = model(images)
                prob_orig = torch.sigmoid(out_orig)

                # 2. Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])
                out_flip, _ = model(images_flipped)
                prob_flip = torch.sigmoid(out_flip)
                prob_flip = torch.flip(prob_flip, dims=[3])

                # Average
                avg_prob = (prob_orig + prob_flip) / 2.0
                avg_prob = avg_prob.cpu().numpy()

                for i, img_id in enumerate(ids):
                    results[img_id] = avg_prob[i, 0]

        return results
