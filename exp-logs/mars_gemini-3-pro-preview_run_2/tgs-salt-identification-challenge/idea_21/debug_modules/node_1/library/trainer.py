import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

from library.config import Config
from library.utils import unpad_image, calc_map_score
from library.losses import SegmentationLoss, DistillationLoss


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaltTrainer:
    """
    Trainer class handling the two-phase training process:
    1. Supervised training of the Privileged Teacher (Image + Depth).
    2. Distillation training of the Multi-Task Student (Image only).
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def train_teacher_epoch(self, model, loader, optimizer, criterion):
        model.train()
        losses = AverageMeter()

        for batch_idx, (images, masks, depths) in enumerate(loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            depths = depths.to(self.device)

            optimizer.zero_grad()

            # Forward pass with depth injection
            logits = model(images, depths)
            loss = criterion(logits, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate_teacher(self, model, loader, criterion):
        model.eval()
        losses = AverageMeter()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks, depths in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                depths = depths.to(self.device)

                logits = model(images, depths)
                loss = criterion(logits, masks)
                losses.update(loss.item(), images.size(0))

                # Post-process for metric calculation
                probs = torch.sigmoid(logits)
                probs = probs.cpu().numpy()
                masks_np = masks.cpu().numpy()

                # Unpad to original size (101x101) for accurate metric calculation
                for i in range(len(probs)):
                    # probs[i] shape is (1, 128, 128)
                    p_unpadded = unpad_image(
                        probs[i, 0], original_shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)
                    )
                    t_unpadded = unpad_image(
                        masks_np[i, 0],
                        original_shape=(Config.ORIG_SIZE, Config.ORIG_SIZE),
                    )

                    all_preds.append(p_unpadded)
                    all_targets.append(t_unpadded)

        # Calculate mAP across the validation set
        map_score = calc_map_score(all_preds, all_targets)
        return losses.avg, map_score

    def fit_teacher(
        self, model, train_loader, val_loader, epochs=Config.TEACHER_EPOCHS
    ):
        """
        Phase 1: Train the Teacher model using Ground Truth Depth.
        """
        print(f"Starting Teacher Training for {epochs} epochs...")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        # Reduce LR when mAP plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=False
        )
        criterion = SegmentationLoss()

        best_map = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_teacher_epoch(
                model, train_loader, optimizer, criterion
            )
            val_loss, val_map = self.validate_teacher(model, val_loader, criterion)

            scheduler.step(val_map)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val mAP: {val_map:.10f}"
            )

            if val_map > best_map:
                best_map = val_map
                patience_counter = 0
                torch.save(model.state_dict(), Config.TEACHER_CHECKPOINT)
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best weights before returning
        if os.path.exists(Config.TEACHER_CHECKPOINT):
            model.load_state_dict(
                torch.load(Config.TEACHER_CHECKPOINT, map_location=self.device)
            )
        return model

    def train_student_epoch(self, student, teacher, loader, optimizer, criterion):
        student.train()
        teacher.eval()  # Teacher is frozen

        losses = AverageMeter()
        seg_losses = AverageMeter()
        dist_losses = AverageMeter()
        depth_losses = AverageMeter()

        for batch_idx, (images, masks, depths) in enumerate(loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            depths = depths.to(self.device)

            # Get Teacher Soft Targets (No Gradient)
            with torch.no_grad():
                teacher_logits = teacher(images, depths)

            optimizer.zero_grad()

            # Student Forward
            student_logits, student_depth_pred = student(images)

            # Calculate Composite Loss
            loss, l_seg, l_distill, l_depth = criterion(
                student_logits, student_depth_pred, teacher_logits, masks, depths
            )

            # Backward
            loss.backward()
            optimizer.step()

            # Update meters
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            seg_losses.update(l_seg.item(), batch_size)
            dist_losses.update(l_distill.item(), batch_size)
            depth_losses.update(l_depth.item(), batch_size)

        return losses.avg, seg_losses.avg, dist_losses.avg, depth_losses.avg

    def validate_student(self, student, loader, criterion):
        student.eval()
        losses = AverageMeter()

        all_preds = []
        all_targets = []

        # For validation, we primarily care about segmentation performance
        val_seg_criterion = SegmentationLoss()

        with torch.no_grad():
            for images, masks, depths in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Student predicts depth internally but we only need segmentation for mAP
                logits, _ = student(images)

                loss = val_seg_criterion(logits, masks.to(self.device))
                losses.update(loss.item(), images.size(0))

                probs = torch.sigmoid(logits)
                probs = probs.cpu().numpy()
                masks_np = masks.cpu().numpy()

                for i in range(len(probs)):
                    p_unpadded = unpad_image(
                        probs[i, 0], original_shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)
                    )
                    t_unpadded = unpad_image(
                        masks_np[i, 0],
                        original_shape=(Config.ORIG_SIZE, Config.ORIG_SIZE),
                    )
                    all_preds.append(p_unpadded)
                    all_targets.append(t_unpadded)

        map_score = calc_map_score(all_preds, all_targets)
        return losses.avg, map_score

    def fit_student(
        self, student, teacher, train_loader, val_loader, epochs=Config.STUDENT_EPOCHS
    ):
        """
        Phase 2: Train the Student model via Distillation and Multi-Task Learning.
        """
        print(f"Starting Student Distillation for {epochs} epochs...")
        optimizer = optim.AdamW(
            student.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=False
        )
        criterion = DistillationLoss()

        best_map = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            t_loss, t_seg, t_dist, t_depth = self.train_student_epoch(
                student, teacher, train_loader, optimizer, criterion
            )
            val_loss, val_map = self.validate_student(student, val_loader, criterion)

            scheduler.step(val_map)

            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {t_loss:.4f} (Seg: {t_seg:.4f}, Dist: {t_dist:.4f}, Depth: {t_depth:.4f}) | Val mAP: {val_map:.10f}"
            )

            if val_map > best_map:
                best_map = val_map
                patience_counter = 0
                torch.save(student.state_dict(), Config.STUDENT_CHECKPOINT)
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best weights
        if os.path.exists(Config.STUDENT_CHECKPOINT):
            student.load_state_dict(
                torch.load(Config.STUDENT_CHECKPOINT, map_location=self.device)
            )
        return student
