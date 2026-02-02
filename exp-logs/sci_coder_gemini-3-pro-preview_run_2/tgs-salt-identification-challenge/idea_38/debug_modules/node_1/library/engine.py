import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from library.config import Config
from library.utils import get_score, unpad_image, rle_encode
from library.losses import TeacherLoss, StudentLoss


class Engine:
    """
    Engine class encapsulating training, validation, and inference logic
    for the FP32-Stabilized Marginalized-Distillation strategy.
    """

    @staticmethod
    def train_teacher_epoch(model, loader, optimizer, device):
        """
        Trains the Specialist Teacher for one epoch using explicit depth injection.
        """
        model.train()
        loss_fn = TeacherLoss()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            images, masks, depths = batch
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            optimizer.zero_grad()

            # Teacher forward pass (requires depth)
            logits = model(images, depths)

            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        return total_loss / n_samples

    @staticmethod
    def train_student_epoch(model, labeled_loader, unlabeled_loader, optimizer, device):
        """
        Trains the Generalist Student for one epoch using labeled data (Supervised)
        and unlabeled data (Distillation with Soft Targets).
        """
        model.train()
        loss_fn = StudentLoss()
        total_loss = 0.0
        n_samples = 0

        # Create an iterator for the unlabeled loader
        unlabeled_iter = iter(unlabeled_loader)

        for labeled_batch in labeled_loader:
            # Get labeled data
            img_lbl, mask_lbl, depth_lbl = labeled_batch
            img_lbl = img_lbl.to(device)
            mask_lbl = mask_lbl.to(device)
            depth_lbl = depth_lbl.to(device)

            # Get unlabeled data (cycle if necessary)
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            img_unlbl, mask_unlbl = unlabeled_batch
            img_unlbl = img_unlbl.to(device)
            mask_unlbl = mask_unlbl.to(device)

            optimizer.zero_grad()

            # --- Labeled Forward & Loss ---
            # Student returns (logits, pred_depth)
            logits_lbl, pred_depth_lbl = model(img_lbl)
            loss_lbl = loss_fn(logits_lbl, pred_depth_lbl, mask_lbl, depth_lbl)

            # --- Unlabeled Forward & Loss ---
            logits_unlbl, pred_depth_unlbl = model(img_unlbl)
            # Unlabeled loss: BCE against soft targets, ignore depth regression
            loss_unlbl = loss_fn(
                logits_unlbl, pred_depth_unlbl, mask_unlbl, depth_targets=None
            )

            # Combine and Backward
            loss = loss_lbl + loss_unlbl
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img_lbl.size(0)
            n_samples += img_lbl.size(0)

        return total_loss / n_samples

    @staticmethod
    def validate(model, loader, device, threshold_range=None):
        """
        Evaluates the model on the validation set and calculates mAP.
        """
        model.eval()
        preds_all = []
        targets_all = []

        with torch.no_grad():
            for batch in loader:
                images, masks, depths = batch
                images = images.to(device)
                depths = depths.to(device)

                # Handle model modes
                if model.mode == "teacher":
                    logits = model(images, depths)
                else:
                    logits, _ = model(images)

                preds = torch.sigmoid(logits)

                # Move to CPU for metric calculation
                preds_np = preds.cpu().numpy()
                masks_np = masks.numpy()

                for i in range(len(preds_np)):
                    # Unpad to original size (101x101)
                    p = unpad_image(preds_np[i, 0], Config.ORIG_SIZE)
                    t = unpad_image(masks_np[i, 0], Config.ORIG_SIZE)
                    preds_all.append(p)
                    targets_all.append(t)

        preds_all = np.array(preds_all)
        targets_all = np.array(targets_all)

        score = get_score(preds_all, targets_all, threshold_range)
        return score

    @staticmethod
    def optimize_threshold(model, loader, device):
        """
        Finds the optimal binarization threshold on the validation set.
        """
        model.eval()
        preds_all = []
        targets_all = []

        with torch.no_grad():
            for batch in loader:
                images, masks, depths = batch
                images = images.to(device)
                depths = depths.to(device)

                if model.mode == "teacher":
                    logits = model(images, depths)
                else:
                    logits, _ = model(images)

                preds = torch.sigmoid(logits).cpu().numpy()
                masks = masks.numpy()

                for i in range(len(preds)):
                    p = unpad_image(preds[i, 0], Config.ORIG_SIZE)
                    t = unpad_image(masks[i, 0], Config.ORIG_SIZE)
                    preds_all.append(p)
                    targets_all.append(t)

        preds_all = np.array(preds_all)
        targets_all = np.array(targets_all)

        # Linear search for best threshold
        thresholds = np.arange(0.3, 0.75, 0.05)
        best_score = -1
        best_th = 0.5

        for th in thresholds:
            # Binarize with current threshold
            binary_preds = (preds_all > th).astype(np.uint8)
            # get_score calculates mAP over IoU thresholds (0.5-0.95)
            score = get_score(binary_preds, targets_all)

            if score > best_score:
                best_score = score
                best_th = th

        return best_th

    @staticmethod
    def generate_marginalized_pseudo_labels(teacher_models, test_loader, device):
        """
        Generates soft pseudo-labels for the test set by marginalizing over depth.
        Returns a dictionary {id: soft_mask_array}.
        """
        for model in teacher_models:
            model.eval()

        scan_depths = Config.MARGINALIZATION_DEPTHS
        pseudo_labels = {}

        with torch.no_grad():
            for batch in test_loader:
                images, _, ids = batch
                images = images.to(device)
                batch_size = images.size(0)

                # Accumulator: (B, 1, H, W)
                batch_probs_sum = torch.zeros(
                    batch_size, 1, Config.PAD_SIZE, Config.PAD_SIZE, device=device
                )

                # Marginalize over depths
                for z_val in scan_depths:
                    # Create constant depth tensor (z-score)
                    d_tensor = torch.full(
                        (batch_size, 1), z_val, device=device, dtype=torch.float32
                    )

                    for model in teacher_models:
                        logits = model(images, d_tensor)
                        probs = torch.sigmoid(logits)
                        batch_probs_sum += probs

                # Average
                total_votes = len(teacher_models) * len(scan_depths)
                batch_avg_probs = batch_probs_sum / total_votes

                # Store
                batch_avg_probs_np = batch_avg_probs.cpu().numpy()
                for i, img_id in enumerate(ids):
                    pseudo_labels[img_id] = batch_avg_probs_np[i]

        return pseudo_labels

    @staticmethod
    def generate_submission(model, loader, device, output_path, threshold=0.5):
        """
        Generates the submission CSV using the Student model with TTA.
        """
        model.eval()
        ids_list = []
        rles_list = []

        with torch.no_grad():
            for batch in loader:
                images, _, ids = batch
                images = images.to(device)

                # TTA: Horizontal Flip
                logits, _ = model(images)
                probs = torch.sigmoid(logits)

                images_flip = torch.flip(images, [3])
                logits_flip, _ = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, [3])

                probs = (probs + probs_flip) / 2.0
                probs_np = probs.cpu().numpy()

                for i, img_id in enumerate(ids):
                    # Unpad
                    pred_mask = unpad_image(probs_np[i, 0], Config.ORIG_SIZE)

                    # Binarize
                    binary_mask = (pred_mask > threshold).astype(np.uint8)

                    # RLE Encode
                    rle = rle_encode(binary_mask)

                    ids_list.append(img_id)
                    rles_list.append(rle)

        # Create DataFrame and save
        sub_df = pd.DataFrame({"id": ids_list, "rle_mask": rles_list})
        sub_df.to_csv(output_path, index=False)

    @staticmethod
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_checkpoint(model, path, device):
        model.load_state_dict(torch.load(path, map_location=device))
