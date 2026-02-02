import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, AverageMeter, dice_score, rle_encode
from library.dataset import ContrailDataset
from library.model import UNetPlusPlus


class DiceLoss(nn.Module):
    """
    Differentiable Dice Loss for training.
    Cite {solution_lesson_node_00010}: Region-based loss is critical for imbalanced segmentation.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Hybrid Loss: BCE + Dice.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class ContrailTrainer:
    """
    Manages the training, validation, checkpointing, and inference lifecycle
    for the Contrail Identification model.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize Model
        self.model = UNetPlusPlus().to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Use Hybrid Loss (Cite {solution_lesson_node_00010})
        self.criterion = CombinedLoss()

        # Checkpoint State
        # List of dictionaries: {'epoch': int, 'dice': float, 'path': str}
        self.top_k_checkpoints = []

    def train_one_epoch(self, loader):
        self.model.train()
        losses = AverageMeter()

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, loader):
        self.model.eval()
        losses = AverageMeter()
        dice_scores = AverageMeter()

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Apply sigmoid for metric calculation
                preds = torch.sigmoid(outputs)
                score = dice_score(preds, masks)

                losses.update(loss.item(), images.size(0))
                dice_scores.update(score, images.size(0))

        return losses.avg, dice_scores.avg

    def save_checkpoint(self, epoch, val_dice):
        """
        Saves checkpoint and manages the Top-K priority queue.
        """
        filename = f"checkpoint_epoch_{epoch}_dice_{val_dice:.6f}.pth"
        path = os.path.join(Config.CHECKPOINT_DIR, filename)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_dice": val_dice,
        }

        torch.save(state, path)

        # Update Top-K List
        self.top_k_checkpoints.append({"epoch": epoch, "dice": val_dice, "path": path})

        # Sort descending by Dice score
        self.top_k_checkpoints.sort(key=lambda x: x["dice"], reverse=True)

        # Enforce Top-K limit
        if len(self.top_k_checkpoints) > Config.CHECKPOINT_TOP_K:
            worst_ckpt = self.top_k_checkpoints.pop()
            if os.path.exists(worst_ckpt["path"]):
                try:
                    os.remove(worst_ckpt["path"])
                except OSError:
                    pass

    def average_weights(self):
        """
        Performs Convergence-Aware Weight Averaging on eligible checkpoints.
        Filters for floating point parameters to protect integer buffers.
        """
        print("\nStarting Convergence-Aware Weight Averaging...")

        # Filter checkpoints saved after the convergence threshold
        eligible_checkpoints = [
            ckpt
            for ckpt in self.top_k_checkpoints
            if ckpt["epoch"] > Config.AVERAGE_START_EPOCH
        ]

        if not eligible_checkpoints:
            print(
                "No checkpoints met the convergence criteria. Using the single best model."
            )
            final_checkpoints = [self.top_k_checkpoints[0]]
        else:
            print(f"Averaging {len(eligible_checkpoints)} checkpoints.")
            final_checkpoints = eligible_checkpoints

        # Load the best model's state dict to serve as the base structure
        # (This preserves integer buffers like num_batches_tracked from the best model)
        best_ckpt_path = self.top_k_checkpoints[0]["path"]
        final_state_dict = torch.load(best_ckpt_path, map_location="cpu")[
            "model_state_dict"
        ]

        if len(final_checkpoints) > 1:
            float_accumulators = {}
            count = 0

            for ckpt in final_checkpoints:
                path = ckpt["path"]
                state = torch.load(path, map_location="cpu")["model_state_dict"]

                for key, value in state.items():
                    if value.is_floating_point():
                        if key not in float_accumulators:
                            float_accumulators[key] = value.clone().double()
                        else:
                            float_accumulators[key] += value.double()
                count += 1

            # Update final_state_dict with averaged float values
            for key, value in float_accumulators.items():
                final_state_dict[key] = (value / count).to(final_state_dict[key].dtype)

        # Save the final averaged model
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        torch.save(final_state_dict, best_model_path)
        print(f"Best model (averaged) saved to {best_model_path}")
        return best_model_path

    def run(self):
        """
        Main execution method.
        """
        # Data Loaders
        train_dataset = ContrailDataset(
            split="train", load_cached_data=True, debug=Config.DEBUG
        )
        val_dataset = ContrailDataset(
            split="validation", load_cached_data=True, debug=Config.DEBUG
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Starting training for {Config.EPOCHS} epochs...")
        print(f"Model: {Config.BACKBONE}, Input Channels: {Config.INPUT_CHANNELS}")

        # Training Loop
        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_dice = self.validate(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Dice: {val_dice}"
            )

            self.save_checkpoint(epoch, val_dice)

        # Weight Averaging
        best_model_path = self.average_weights()

        # Inference
        self.inference(best_model_path)

    def inference(self, model_path):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        print("Starting Inference on Test Set...")

        # Load Averaged Model
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Test Loader
        test_dataset = ContrailDataset(
            split="test", load_cached_data=True, debug=Config.DEBUG
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        results = []

        with torch.no_grad():
            for images, record_ids in test_loader:
                images = images.to(self.device)

                # TTA: Test Time Augmentation (Cite {solution_lesson_node_00016})

                # 1. Original
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                logits_h = self.model(images_h)
                probs_h = torch.flip(torch.sigmoid(logits_h), [3])

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                logits_v = self.model(images_v)
                probs_v = torch.flip(torch.sigmoid(logits_v), [2])

                # Average Predictions
                avg_probs = (probs + probs_h + probs_v) / 3.0

                # Thresholding
                preds = (avg_probs > 0.5).float().cpu().numpy()

                # Encode
                for i in range(len(preds)):
                    # preds shape is (B, 1, H, W), take (H, W)
                    mask = preds[i, 0]
                    rle = rle_encode(mask)
                    results.append({"record_id": record_ids[i], "encoded_pixels": rle})

        # Save Submission
        df = pd.DataFrame(results)
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")


def run_training():
    """
    Entry point for the training module.
    """
    trainer = ContrailTrainer()
    trainer.run()
