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
    def train_epoch(model, loader, optimizer, device, loss_fn, scheduler=None):
        """
        Trains the model for one epoch.
        """
        model.train()
        running_loss = 0.0

        for batch in loader:
            images, masks, depths, _ = batch

            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            # Depth Jitter: Add Gaussian noise to depth to prevent overfitting
            if Config.DEPTH_JITTER_STD > 0:
                noise = torch.randn_like(depths) * Config.DEPTH_JITTER_STD
                depths = depths + noise

            optimizer.zero_grad()

            # Forward: Requires image and depth
            outputs = model(images, depths)

            loss = loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        return running_loss / len(loader)

    @staticmethod
    def validate(model, loader, device, loss_fn):
        """
        Evaluates the model on the validation set.
        Performs dynamic threshold optimization to find the best mAP.
        """
        model.eval()
        running_loss = 0.0

        # Store predictions and targets for threshold optimization
        all_preds = []
        all_masks = []

        with torch.no_grad():
            for batch in loader:
                images, masks, depths, _ = batch

                images = images.to(device)
                masks = masks.to(device)
                depths = depths.to(device)

                outputs = model(images, depths)
                loss = loss_fn(outputs, masks)

                probs = torch.sigmoid(outputs)

                running_loss += loss.item()

                all_preds.append(probs.cpu().numpy())
                all_masks.append(masks.cpu().numpy())

        # Concatenate
        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Threshold Optimization
        # We sweep binarization thresholds to find the best competition metric score
        best_score = 0.0
        thresholds = np.arange(0.3, 0.75, 0.05)

        for t in thresholds:
            # Binarize
            bin_preds = (all_preds > t).astype(np.uint8)
            # Calculate batch score
            score = get_batch_iou_score(bin_preds, all_masks)
            if score > best_score:
                best_score = score

        return running_loss / len(loader), best_score

    @staticmethod
    def predict_tta(model, loader, device):
        """
        Inference with Test-Time Augmentation (Horizontal Flip).
        """
        model.eval()
        results = {}

        with torch.no_grad():
            for batch in loader:
                images, depths, ids = batch
                images = images.to(device)
                depths = depths.to(device)

                # 1. Original
                out_orig = model(images, depths)
                prob_orig = torch.sigmoid(out_orig)

                # 2. Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])
                out_flip = model(images_flipped, depths)
                prob_flip = torch.sigmoid(out_flip)
                prob_flip = torch.flip(prob_flip, dims=[3])

                # Average
                avg_prob = (prob_orig + prob_flip) / 2.0
                avg_prob = avg_prob.cpu().numpy()

                for i, img_id in enumerate(ids):
                    results[img_id] = avg_prob[i, 0]

        return results
