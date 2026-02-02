import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.utils import set_seed, do_kaggle_metric
from library.model import DeepResUNet
from library.loss import CompoundLoss
from library.dataset import get_loaders


class Trainer:
    def __init__(
        self,
        epochs=150,
        batch_size=32,
        num_workers=4,
        lr=1e-3,
        weight_decay=1e-4,
        checkpoint_dir="./working/checkpoints",
        cache_dir="./working/idea_10",
        debug=False,
    ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir
        self.cache_dir = cache_dir
        self.debug = debug

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        set_seed(42)

    def fit(self):
        # 1. Data Loading
        print("Initializing Data Loaders...")
        train_loader, val_loader = get_loaders(
            train_metadata_path="./metadata/train.csv",
            val_metadata_path="./metadata/val.csv",
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            load_cached_data=True,
            debug=self.debug,
        )

        # 2. Model Setup
        print("Initializing Model...")
        model = DeepResUNet(in_channels=2, classes=1).to(self.device)

        # 3. Optimization
        optimizer = AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Schedule: 3 cycles of 50 epochs
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=1e-5
        )

        # 4. Losses
        criterion_main = CompoundLoss()
        criterion_aux = nn.BCEWithLogitsLoss()

        best_map = 0.0

        print(f"Starting training for {self.epochs} epochs on {self.device}...")

        for epoch in range(self.epochs):
            start_time = time.time()

            # --- Training Step ---
            model.train()
            train_loss_meter = 0.0

            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()

                outputs = model(images)

                # Resize masks for auxiliary heads (nearest neighbor for binary masks)
                mask_32 = F.interpolate(masks, size=(32, 32), mode="nearest")
                mask_64 = F.interpolate(masks, size=(64, 64), mode="nearest")

                # Calculate Losses
                # CompoundLoss handles seg + cls internally
                loss_compound = criterion_main(outputs, masks)

                # Aux Losses
                loss_aux1 = criterion_aux(outputs["aux_64"], mask_64)
                loss_aux2 = criterion_aux(outputs["aux_32"], mask_32)

                # Total Loss with Deep Supervision weights
                loss = loss_compound + 0.5 * loss_aux1 + 0.5 * loss_aux2

                loss.backward()
                optimizer.step()

                train_loss_meter += loss.item()

            # Update Scheduler at end of epoch
            scheduler.step()

            avg_train_loss = train_loss_meter / len(train_loader)

            # --- Validation Step ---
            model.eval()
            val_loss_meter = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = model(images)

                    # Validation Loss Calculation
                    mask_32 = F.interpolate(masks, size=(32, 32), mode="nearest")
                    mask_64 = F.interpolate(masks, size=(64, 64), mode="nearest")

                    loss_compound = criterion_main(outputs, masks)
                    loss_aux1 = criterion_aux(outputs["aux_64"], mask_64)
                    loss_aux2 = criterion_aux(outputs["aux_32"], mask_32)

                    loss = loss_compound + 0.5 * loss_aux1 + 0.5 * loss_aux2
                    val_loss_meter += loss.item()

                    # Inference Logic for Metric
                    logits = outputs["logits"]
                    final_probs = torch.sigmoid(logits)

                    # Unpad: 128x128 -> 101x101
                    # Padding was: Top=13, Bottom=14, Left=13, Right=14
                    # Indices: [13 : 128-14] -> [13 : 114]
                    final_probs_cropped = final_probs[:, :, 13:114, 13:114]
                    masks_cropped = masks[:, :, 13:114, 13:114]

                    all_preds.append(final_probs_cropped.cpu().numpy())
                    all_targets.append(masks_cropped.cpu().numpy())

            avg_val_loss = val_loss_meter / len(val_loader)

            # Concatenate and Metric
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            # do_kaggle_metric expects (N, H, W) usually, but handles (N, 1, H, W) if numpy
            # Ensure shape is correct (squeeze channel)
            if all_preds.ndim == 4:
                all_preds = all_preds.squeeze(1)
            if all_targets.ndim == 4:
                all_targets = all_targets.squeeze(1)

            current_map = do_kaggle_metric(all_preds, all_targets, threshold=0.5)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{self.epochs} | Time: {elapsed:.1f}s | "
                f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                f"Val mAP: {current_map}"
            )

            # --- Checkpointing ---

            # 1. Best Model (based on mAP)
            if current_map > best_map:
                best_map = current_map
                torch.save(
                    model.state_dict(),
                    os.path.join(self.checkpoint_dir, "best_model.pth"),
                )

            # 2. Snapshot Ensembling (End of Cycle 2 -> Epoch 100)
            if (epoch + 1) == 100:
                print("Saving Cycle 2 Snapshot...")
                torch.save(
                    model.state_dict(),
                    os.path.join(self.checkpoint_dir, "best_cycle_2.pth"),
                )

            # 3. Snapshot Ensembling (End of Cycle 3 -> Epoch 150)
            if (epoch + 1) == 150:
                print("Saving Cycle 3 Snapshot...")
                torch.save(
                    model.state_dict(),
                    os.path.join(self.checkpoint_dir, "best_cycle_3.pth"),
                )

        print(f"Training complete. Best Validation mAP: {best_map}")


def train_model(epochs=150, batch_size=32, debug=False):
    trainer = Trainer(epochs=epochs, batch_size=batch_size, debug=debug)
    trainer.fit()
