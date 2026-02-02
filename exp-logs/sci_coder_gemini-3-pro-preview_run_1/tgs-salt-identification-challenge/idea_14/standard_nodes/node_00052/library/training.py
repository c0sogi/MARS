import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

from library.utils import set_seed, calculate_iou_map, rle_encode
from library.losses import CurriculumLoss
from library.model import DeepResUNet
from library.dataset import get_dataloaders, TARGET_H, TARGET_W, ORIG_H, ORIG_W

# Configuration Constants
CHECKPOINT_DIR = "./working/idea_14/checkpoints"
SUBMISSION_DIR = "./working/idea_14/submission"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss with Intra-Cycle Curriculum
        self.criterion = CurriculumLoss()

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )

        # Scheduler: Cosine Annealing with Warm Restarts
        # T_0=50 aligns with the 50-epoch cycle length defined in the strategy
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=1, eta_min=1e-5
        )

        # State tracking
        self.best_map_c2 = -1.0
        self.best_map_c3 = -1.0

        # Ensure directories exist
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for images, masks, depths, _ in self.train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            depths = depths.to(DEVICE)

            self.optimizer.zero_grad()

            # Forward pass: returns list [logits_128, logits_64, logits_32] due to Deep Supervision
            outputs = self.model(images, depths)

            # Calculate Loss for Main Head (128x128)
            loss = self.criterion(outputs[0], masks, epoch)

            # Calculate Loss for Aux Head 1 (64x64)
            masks_64 = F.interpolate(masks, size=(64, 64), mode="nearest")
            loss += 0.5 * self.criterion(outputs[1], masks_64, epoch)

            # Calculate Loss for Aux Head 2 (32x32)
            masks_32 = F.interpolate(masks, size=(32, 32), mode="nearest")
            loss += 0.25 * self.criterion(outputs[2], masks_32, epoch)

            # Optimization Step
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        # Update Scheduler at the end of epoch
        self.scheduler.step()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        preds_list = []
        masks_list = []

        # Calculate cropping indices to restore 101x101 resolution
        start_h = (TARGET_H - ORIG_H) // 2
        start_w = (TARGET_W - ORIG_W) // 2

        with torch.no_grad():
            for images, masks, depths, _ in self.val_loader:
                images = images.to(DEVICE)
                depths = depths.to(DEVICE)

                # Forward pass in eval mode returns only the final logits (128x128)
                logits = self.model(images, depths)
                probs = torch.sigmoid(logits)

                # Move to CPU for metric calculation
                probs_np = probs.cpu().numpy()
                masks_np = masks.numpy()

                # Center Crop to original dimensions (101x101)
                probs_cropped = probs_np[
                    :, 0, start_h : start_h + ORIG_H, start_w : start_w + ORIG_W
                ]
                masks_cropped = masks_np[
                    :, 0, start_h : start_h + ORIG_H, start_w : start_w + ORIG_W
                ]

                preds_list.append(probs_cropped)
                masks_list.append(masks_cropped)

        # Concatenate all batches
        all_preds = np.concatenate(preds_list, axis=0)
        all_masks = np.concatenate(masks_list, axis=0)

        # Threshold predictions for IoU calculation
        binary_preds = (all_preds > 0.5).astype(np.uint8)
        binary_masks = (all_masks > 0.5).astype(np.uint8)

        # Calculate mAP
        map_score = calculate_iou_map(binary_preds, binary_masks)
        return map_score

    def run(self, epochs=150):
        print(f"Starting training for {epochs} epochs on {DEVICE}...")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(epoch)
            val_map = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f} - Val mAP: {val_map}"
            )

            # Checkpointing Logic based on Cycles
            # Cycle 2: Epochs 50 to 99
            if 50 <= epoch < 100:
                if val_map > self.best_map_c2:
                    self.best_map_c2 = val_map
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(CHECKPOINT_DIR, "best_cycle_2.pth"),
                    )
                    print(f"Saved Best Cycle 2 Model (mAP: {self.best_map_c2})")

            # Cycle 3: Epochs 100 to 149
            elif 100 <= epoch < 150:
                if val_map > self.best_map_c3:
                    self.best_map_c3 = val_map
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(CHECKPOINT_DIR, "best_cycle_3.pth"),
                    )
                    print(f"Saved Best Cycle 3 Model (mAP: {self.best_map_c3})")

    def predict_test_set(self):
        print("Starting Inference and Submission Generation...")

        # Paths to best models
        path_c2 = os.path.join(CHECKPOINT_DIR, "best_cycle_2.pth")
        path_c3 = os.path.join(CHECKPOINT_DIR, "best_cycle_3.pth")

        # Initialize models
        model_c2 = DeepResUNet().to(DEVICE)
        model_c3 = DeepResUNet().to(DEVICE)

        # Load weights if available
        c2_loaded = False
        if os.path.exists(path_c2):
            model_c2.load_state_dict(torch.load(path_c2, map_location=DEVICE))
            c2_loaded = True

        c3_loaded = False
        if os.path.exists(path_c3):
            model_c3.load_state_dict(torch.load(path_c3, map_location=DEVICE))
            c3_loaded = True

        if not c2_loaded and not c3_loaded:
            print("Warning: No checkpoints found. Using current model state.")
            model_c2.load_state_dict(self.model.state_dict())
            model_c3.load_state_dict(self.model.state_dict())
            self.best_map_c2 = 0.5  # Dummy values to proceed
            self.best_map_c3 = 0.5

        model_c2.eval()
        model_c3.eval()

        # Quality-Gated Ensembling Logic
        # If the difference in performance is small (< 0.005), ensemble to reduce variance.
        # Otherwise, trust the significantly better model.
        diff = abs(self.best_map_c2 - self.best_map_c3)
        use_ensemble = diff < 0.005

        # Determine best single model
        best_single_model = (
            model_c2 if self.best_map_c2 >= self.best_map_c3 else model_c3
        )

        print(f"Cycle 2 Best mAP: {self.best_map_c2}")
        print(f"Cycle 3 Best mAP: {self.best_map_c3}")
        print(
            f"Strategy: {'Ensemble (Average)' if use_ensemble else 'Best Single Model'}"
        )

        predictions = []
        ids = []

        start_h = (TARGET_H - ORIG_H) // 2
        start_w = (TARGET_W - ORIG_W) // 2

        with torch.no_grad():
            for images, _, depths, img_ids in self.test_loader:
                images = images.to(DEVICE)
                depths = depths.to(DEVICE)

                # TTA Helper: Predict -> Flip -> Predict -> FlipBack -> Average
                def predict_tta(model, x, z):
                    # Original
                    pred = torch.sigmoid(model(x, z))
                    # Horizontal Flip
                    x_flip = torch.flip(x, [3])
                    pred_flip = torch.sigmoid(model(x_flip, z))
                    pred_flip_back = torch.flip(pred_flip, [3])
                    return (pred + pred_flip_back) / 2.0

                if use_ensemble:
                    p2 = predict_tta(model_c2, images, depths)
                    p3 = predict_tta(model_c3, images, depths)
                    avg_pred = (p2 + p3) / 2.0
                else:
                    avg_pred = predict_tta(best_single_model, images, depths)

                # Crop to 101x101
                avg_pred = avg_pred[
                    :, 0, start_h : start_h + ORIG_H, start_w : start_w + ORIG_W
                ]

                # Threshold at 0.5
                binary_masks = (avg_pred > 0.5).cpu().numpy().astype(np.uint8)

                # Encode and store
                for i in range(len(img_ids)):
                    rle = rle_encode(binary_masks[i])
                    predictions.append(rle)
                    ids.append(img_ids[i])

        # Save Submission
        df = pd.DataFrame({"id": ids, "rle_mask": predictions})
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")


def run_training(epochs=150, batch_size=64, load_cached_data=True):
    """
    Main entry point for the training module.
    """
    set_seed(42)

    # Load Data
    # load_cached_data flag is passed to handle caching requirements
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data, num_workers=4
    )

    # Initialize Model
    model = DeepResUNet(in_channels=1, out_channels=1)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader)

    # Execute Training Pipeline
    trainer.run(epochs=epochs)

    # Execute Inference Pipeline
    trainer.predict_test_set()
