import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.utils import set_seed, do_kaggle_metric, unpad_image_101, rle_encode
from library.losses import get_loss_for_phase, BCEDiceLoss
from library.model import ResUNetPPM
from library.dataset import get_dataloaders


class Trainer:
    def __init__(
        self,
        base_dir="./working/idea_6",
        batch_size=32,
        num_workers=2,
        lr=1e-3,
        epochs=150,
        device=None,
        debug=False,
    ):
        """
        Args:
            base_dir (str): Directory to save checkpoints and logs.
            batch_size (int): Batch size for training.
            num_workers (int): Number of worker threads for data loading.
            lr (float): Learning rate.
            epochs (int): Total training epochs.
            device (str): 'cuda' or 'cpu'. Auto-detected if None.
            debug (bool): If True, runs for limited steps for debugging.
        """
        self.base_dir = base_dir
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.epochs = epochs
        self.debug = debug

        # Cycle configuration for CosineAnnealingWarmRestarts
        # 3 cycles of 50 epochs
        self.cycle_len = 50

        set_seed(42)

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            load_cached_data=True,
        )

        # Model
        self.model = ResUNetPPM(in_channels=2, num_classes=1, filters=64).to(
            self.device
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )

        # Scheduler: Cosine Annealing Warm Restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.cycle_len, T_mult=1, eta_min=1e-6
        )

        # Auxiliary Loss (Always BCE+Dice for deep supervision heads)
        self.aux_criterion = BCEDiceLoss().to(self.device)

        # Metric Tracking
        self.best_map_global = 0.0
        self.best_map_cycle_2 = 0.0
        self.best_map_cycle_3 = 0.0

    def train_epoch(self, epoch):
        self.model.train()

        # Get main loss function based on phase (BCE+Dice -> Lovasz)
        # Switch happens at epoch 100 (start of Cycle 3)
        criterion = get_loss_for_phase(epoch, switch_epoch=100).to(self.device)

        running_loss = 0.0
        count = 0

        for i, (images, masks, _) in enumerate(self.train_loader):
            if self.debug and i >= 5:
                break

            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: returns logits and aux outputs
            logits, aux1, aux2, aux3 = self.model(images)

            # Main Loss
            loss_main = criterion(logits, masks)

            # Aux Losses (Deep Supervision)
            # We use a weight of 0.5 for auxiliary losses
            loss_aux1 = self.aux_criterion(aux1, masks)
            loss_aux2 = self.aux_criterion(aux2, masks)
            loss_aux3 = self.aux_criterion(aux3, masks)

            total_loss = loss_main + 0.5 * (loss_aux1 + loss_aux2 + loss_aux3)

            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item() * images.size(0)
            count += images.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate_epoch(self):
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (images, masks, _) in enumerate(self.val_loader):
                if self.debug and i >= 5:
                    break

                images = images.to(self.device)
                # In eval mode, model returns only logits
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # Move to CPU numpy for unpadding and metric calculation
                probs_np = probs.squeeze(1).cpu().numpy()  # (B, 128, 128)
                masks_np = masks.squeeze(1).cpu().numpy()  # (B, 128, 128)

                # Unpad each image in the batch to restore 101x101 resolution
                for b in range(probs_np.shape[0]):
                    p_unpad = unpad_image_101(probs_np[b])
                    t_unpad = unpad_image_101(masks_np[b])
                    all_preds.append(p_unpad)
                    all_targets.append(t_unpad)

        if not all_preds:
            return 0.0

        # Stack into arrays (N, 101, 101)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate Kaggle Metric (mAP over IoU thresholds)
        score = do_kaggle_metric(all_preds, all_targets)
        return score

    def save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)

    def fit(self):
        print(f"Starting training on {self.device} for {self.epochs} epochs.")
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch(epoch)
            val_map = self.validate_epoch()

            # Step scheduler
            self.scheduler.step()

            # Snapshot Ensembling Logic
            # Cycle 1: Epochs 0-49
            # Cycle 2: Epochs 50-99
            # Cycle 3: Epochs 100-149

            # Save Global Best
            if val_map > self.best_map_global:
                self.best_map_global = val_map
                self.save_checkpoint("best_model.pth")

            # Save Cycle 2 Best
            if 50 <= epoch < 100:
                if val_map > self.best_map_cycle_2:
                    self.best_map_cycle_2 = val_map
                    self.save_checkpoint("best_cycle_2.pth")

            # Save Cycle 3 Best
            if 100 <= epoch < 150:
                if val_map > self.best_map_cycle_3:
                    self.best_map_cycle_3 = val_map
                    self.save_checkpoint("best_cycle_3.pth")

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val mAP: {val_map:.10f} | "
                f"Time: {time.time() - epoch_start:.2f}s"
            )

            if self.debug and epoch >= 2:
                print("Debug mode: stopping early.")
                break

        print(f"Training finished. Total time: {time.time() - start_time:.2f}s")
        print(f"Best Global mAP: {self.best_map_global:.10f}")
        print(f"Best Cycle 2 mAP: {self.best_map_cycle_2:.10f}")
        print(f"Best Cycle 3 mAP: {self.best_map_cycle_3:.10f}")

    def generate_submission(self):
        """
        Generates submission.csv by ensembling the best models from Cycle 2 and Cycle 3.
        Applies Test-Time Augmentation (Horizontal Flip).
        """
        print("Generating submission...")

        # Define checkpoints to ensemble
        checkpoints = ["best_cycle_2.pth", "best_cycle_3.pth"]
        valid_checkpoints = []

        for cp in checkpoints:
            path = os.path.join(self.checkpoint_dir, cp)
            if os.path.exists(path):
                valid_checkpoints.append(path)

        # Fallback if specific cycle checkpoints are missing
        if not valid_checkpoints:
            print("Cycle checkpoints not found. Falling back to best_model.pth")
            fallback = os.path.join(self.checkpoint_dir, "best_model.pth")
            if os.path.exists(fallback):
                valid_checkpoints.append(fallback)
            else:
                print("Error: No checkpoints found.")
                return

        results = {}  # id -> accumulated_probability_sum

        # Iterate over each model in the ensemble
        for cp_path in valid_checkpoints:
            print(f"Ensembling model: {cp_path}")
            self.model.load_state_dict(torch.load(cp_path, map_location=self.device))
            self.model.eval()

            with torch.no_grad():
                for i, (images, img_ids) in enumerate(self.test_loader):
                    if self.debug and i >= 5:
                        break

                    images = images.to(self.device)

                    # 1. Standard Prediction
                    logits = self.model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, 128, 128)

                    # 2. TTA: Horizontal Flip
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = self.model(images_flipped)
                    probs_flipped = torch.sigmoid(logits_flipped).cpu().numpy()
                    # Flip back to original orientation
                    probs_flipped = np.flip(probs_flipped, axis=3)

                    # Average TTA
                    avg_probs = (probs + probs_flipped) / 2.0

                    # Accumulate results
                    for b, img_id in enumerate(img_ids):
                        # Unpad to 101x101
                        prob_map = avg_probs[b, 0]  # (128, 128)
                        prob_unpad = unpad_image_101(prob_map)  # (101, 101)

                        if img_id not in results:
                            results[img_id] = np.zeros_like(
                                prob_unpad, dtype=np.float32
                            )

                        results[img_id] += prob_unpad

        # Finalize and Save
        submission_data = []
        num_models = len(valid_checkpoints)

        for img_id, prob_sum in results.items():
            # Average over ensemble
            final_prob = prob_sum / num_models

            # Threshold at 0.5
            binary_mask = (final_prob > 0.5).astype(np.uint8)

            # RLE Encode
            rle = rle_encode(binary_mask)
            submission_data.append({"id": img_id, "rle_mask": rle})

        df_sub = pd.DataFrame(submission_data)

        # Ensure output directory exists
        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        output_path = os.path.join(sub_dir, "submission.csv")

        df_sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
