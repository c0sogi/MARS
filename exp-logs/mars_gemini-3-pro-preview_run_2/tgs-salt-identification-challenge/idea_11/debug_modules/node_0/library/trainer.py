import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import library modules
from library.model import FiLMResNet34
from library.loss import BCELovaszLoss
from library.dataset import SaltDataset
from library.utils import rle_encode, optimize_threshold


class SaltTrainer:
    def __init__(
        self,
        device_name="cuda",
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-2,
        epochs=50,
        patience=10,
        seed=42,
    ):

        self.seed = seed
        self.set_seed(self.seed)

        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # Directories
        self.working_dir = "./working"
        self.submission_dir = "./submission"
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.working_dir, "best_model.pth")

        # Data Loaders
        # Note: SaltDataset handles caching and depth masking internally
        self.train_dataset = SaltDataset(mode="train", load_cached=True)
        self.val_dataset = SaltDataset(mode="val", load_cached=True)
        self.test_dataset = SaltDataset(mode="test", load_cached=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Model
        self.model = FiLMResNet34(num_classes=1, pretrained=True)
        self.model.to(self.device)

        # Loss
        self.criterion = BCELovaszLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        for images, masks, depths, _ in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            depths = depths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # Depth masking is handled by dataset, so we just pass depths
            logits = self.model(images, depths)

            loss = self.criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks, depths, _ in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                depths = depths.to(self.device)

                logits = self.model(images, depths)
                loss = self.criterion(logits, masks)

                running_loss += loss.item() * images.size(0)

                # Apply sigmoid to get probabilities
                preds = torch.sigmoid(logits)

                # Crop back to 101x101 for metric calculation
                # Model output is 128x128. Center crop.
                # Pad logic was: 101 -> 128 (diff 27). Top 13, Bottom 14.
                preds_cropped = preds[:, :, 13:114, 13:114]
                masks_cropped = masks[:, :, 13:114, 13:114]

                all_preds.append(preds_cropped.cpu())
                all_targets.append(masks_cropped.cpu())

        epoch_loss = running_loss / len(self.val_dataset)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Optimize threshold
        best_threshold, best_map = optimize_threshold(all_preds, all_targets)

        return epoch_loss, best_threshold, best_map

    def train(self):
        print(f"Starting training on {self.device}...")
        best_map = 0.0
        patience_counter = 0
        best_threshold = 0.5

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss, threshold, val_map = self.validate()

            # Step scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val mAP: {val_map:.6f} | "
                f"Best Thresh: {threshold:.4f}"
            )

            # Checkpointing and Early Stopping
            if val_map > best_map:
                best_map = val_map
                best_threshold = threshold
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  -> Model saved! New best mAP: {best_map:.6f}")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(
            f"Training complete. Best mAP: {best_map:.6f} at threshold {best_threshold:.4f}"
        )
        return best_threshold

    def predict_test(self, threshold):
        print("Generating test predictions...")

        # Load best model
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
        else:
            print("Warning: No checkpoint found, using current model weights.")

        self.model.eval()
        submission_data = []

        with torch.no_grad():
            for images, _, depths, ids in self.test_loader:
                images = images.to(self.device)
                depths = depths.to(self.device)

                # TTA: Original
                logits_orig = self.model(images, depths)
                probs_orig = torch.sigmoid(logits_orig)

                # TTA: Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = self.model(images_flipped, depths)
                probs_flipped = torch.sigmoid(logits_flipped)
                probs_flipped = torch.flip(probs_flipped, dims=[3])

                # Average
                probs_avg = (probs_orig + probs_flipped) / 2.0

                # Crop to 101x101
                # Indices: 13 to 114 (13+101)
                probs_cropped = probs_avg[:, :, 13:114, 13:114]

                # Process batch
                probs_np = probs_cropped.cpu().numpy()

                for i in range(len(ids)):
                    img_id = ids[i]
                    prob_map = probs_np[i, 0, :, :]

                    # Binarize
                    mask = (prob_map > threshold).astype(np.uint8)

                    # RLE Encode
                    rle = rle_encode(mask)
                    submission_data.append([img_id, rle])

        # Save submission
        sub_df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        out_path = os.path.join(self.submission_dir, "submission.csv")
        sub_df.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")


def main():
    # Initialize trainer
    trainer = SaltTrainer(epochs=50, patience=10)

    # Train and find best threshold
    best_threshold = trainer.train()

    # Generate submission
    trainer.predict_test(best_threshold)
