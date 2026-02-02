import os
import gc
import collections
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
import cv2
from scipy.ndimage import label

from library.config import Config
from library.utils import set_seed, compute_metrics, rle_encode
from library.dataset import prepare_data, UWMadisonDataset, get_transforms
from library.model import SegFormer
from library.loss import ComboLoss


class Trainer:
    """
    Trainer class for the Lightweight SegFormer model.
    Handles training, validation, and inference pipelines.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        set_seed(config.SEED)

        # Initialize Model
        self.model = SegFormer(
            backbone_name=config.BACKBONE,
            num_classes=config.NUM_CLASSES,
            pretrained=True,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Function
        self.criterion = ComboLoss()

        # Scheduler (initialized in fit method based on dataset size)
        self.scheduler = None

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = len(loader.dataset)

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Step scheduler per batch
            if self.scheduler:
                self.scheduler.step()

            running_loss += loss.item() * images.size(0)

        return running_loss / dataset_size

    def validate(self, loader):
        """
        Runs validation and computes 3D metrics.
        Aggregates 2D slice predictions into 3D volumes to calculate Hausdorff distance.
        """
        self.model.eval()
        volume_data = collections.defaultdict(list)

        # Inference on validation set
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                ids = batch["id"]

                outputs = self.model(images)
                # Apply sigmoid and threshold
                preds = (torch.sigmoid(outputs) > 0.5).float()

                preds = preds.cpu().numpy()
                masks = masks.cpu().numpy()

                for i, img_id in enumerate(ids):
                    # Parse ID: caseXXX_dayYY_slice_ZZZZ
                    parts = img_id.split("_")
                    case_day = f"{parts[0]}_{parts[1]}"
                    slice_idx = int(parts[3])

                    volume_data[case_day].append(
                        {
                            "slice": slice_idx,
                            "pred": preds[i],  # Shape: (C, H, W)
                            "gt": masks[i],  # Shape: (C, H, W)
                        }
                    )

        # Compute 3D Metrics
        dice_scores = []
        hd_scores = []

        # Iterate over each reconstructed volume
        for case_day, slices in volume_data.items():
            # Sort slices by index to ensure correct Z-ordering
            slices.sort(key=lambda x: x["slice"])

            # Stack slices: List of (C, H, W) -> (D, C, H, W) -> Transpose to (C, D, H, W)
            vol_pred = np.stack([s["pred"] for s in slices], axis=1)
            vol_gt = np.stack([s["gt"] for s in slices], axis=1)

            # Calculate metrics for each class
            for c in range(self.config.NUM_CLASSES):
                p = vol_pred[c]
                g = vol_gt[c]

                metrics = compute_metrics(p, g)
                dice_scores.append(metrics["dice"])
                hd_scores.append(metrics["hausdorff"])

        mean_dice = np.mean(dice_scores) if dice_scores else 0.0
        mean_hd = np.mean(hd_scores) if hd_scores else 0.0

        # Combined Score: 0.4 * Dice + 0.6 * (1 - Hausdorff)
        # Note: Hausdorff is a distance (lower is better), so we use (1 - HD) for scoring.
        score = 0.4 * mean_dice + 0.6 * (1.0 - mean_hd)

        return score, mean_dice, mean_hd

    def fit(self, epochs=None, debug=False):
        """
        Main training loop with Early Stopping and Model Checkpointing.
        """
        if epochs is None:
            epochs = self.config.EPOCHS

        print("Preparing data...")
        # Load and process metadata (cached if available)
        train_df = prepare_data(self.config.TRAIN_METADATA_PATH, mode="train")
        val_df = prepare_data(self.config.VAL_METADATA_PATH, mode="val")

        if debug:
            print("Debug mode: Subsampling dataset.")
            train_df = train_df.sample(
                n=min(len(train_df), 200), random_state=self.config.SEED
            ).reset_index(drop=True)
            val_df = val_df.sample(
                n=min(len(val_df), 50), random_state=self.config.SEED
            ).reset_index(drop=True)

        # Create Datasets and Loaders
        train_dataset = UWMadisonDataset(
            train_df, get_transforms("train"), mode="train"
        )
        val_dataset = UWMadisonDataset(val_df, get_transforms("val"), mode="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Setup Scheduler: Linear Warmup -> Polynomial Decay
        warmup_epochs = 1
        total_iters = len(train_loader) * epochs
        warmup_iters = len(train_loader) * warmup_epochs

        scheduler1 = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters
        )
        scheduler2 = PolynomialLR(
            self.optimizer, total_iters=total_iters - warmup_iters, power=1.0
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[warmup_iters],
        )

        print(f"Starting training for {epochs} epochs...")

        best_score = -np.inf
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_score, val_dice, val_hd = self.validate(val_loader)

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.6f} | Dice: {val_dice:.6f} | HD: {val_hd:.6f}"
            )

            if val_score > best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"  >>> Best model saved (Score: {best_score:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training finished. Best Validation Score: {best_score:.6f}")

        # Cleanup
        del train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()

    def inference(self):
        """
        Generates predictions for the test set.
        Applies 3D connected component analysis and generates RLE submission.
        """
        print("Starting Inference...")

        # Load test data
        test_df = prepare_data(self.config.TEST_METADATA_PATH, mode="test")
        test_dataset = UWMadisonDataset(test_df, get_transforms("test"), mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
        )

        # Load best model
        if not os.path.exists(self.config.MODEL_SAVE_PATH):
            print("No trained model found. Skipping inference.")
            return

        self.model.load_state_dict(
            torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        # Storage for 3D processing
        case_data = collections.defaultdict(list)

        print("Running prediction loop...")
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                ids = batch["id"]
                orig_h = batch["orig_h"].numpy()
                orig_w = batch["orig_w"].numpy()

                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().cpu().numpy()

                for i, img_id in enumerate(ids):
                    parts = img_id.split("_")
                    case_day = f"{parts[0]}_{parts[1]}"
                    slice_idx = int(parts[3])

                    case_data[case_day].append(
                        {
                            "slice": slice_idx,
                            "pred": preds[i],  # (C, 256, 256)
                            "id": img_id,
                            "h": orig_h[i],
                            "w": orig_w[i],
                        }
                    )

        # Post-processing and Encoding
        results = []
        class_names = ["large_bowel", "small_bowel", "stomach"]

        print("Post-processing volumes and generating RLE...")
        for case_day, slices in case_data.items():
            slices.sort(key=lambda x: x["slice"])

            # Reconstruct 3D Volume: (C, D, H_small, W_small)
            vol_small = np.stack([s["pred"] for s in slices], axis=1)

            # 3D Connected Component Analysis: Keep largest component per class
            for c in range(self.config.NUM_CLASSES):
                class_vol = vol_small[c]  # (D, H, W)
                labeled_vol, num_features = label(class_vol)
                if num_features > 1:
                    # Find largest component (background is 0)
                    sizes = [
                        np.sum(labeled_vol == k) for k in range(1, num_features + 1)
                    ]
                    largest_k = np.argmax(sizes) + 1
                    # Keep only largest
                    class_vol = (labeled_vol == largest_k).astype(float)
                    vol_small[c] = class_vol

            # Resize back to original dimensions and encode
            for i, s_info in enumerate(slices):
                slice_pred_small = vol_small[:, i, :, :]  # (C, 256, 256)
                orig_h, orig_w = s_info["h"], s_info["w"]

                rles = []
                for c in range(self.config.NUM_CLASSES):
                    mask_small = slice_pred_small[c]
                    # Resize to original (W, H) for cv2
                    mask_orig = cv2.resize(
                        mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                    rle = rle_encode(mask_orig)
                    rles.append(rle)

                # Append results
                for c_idx, c_name in enumerate(class_names):
                    results.append(
                        {"id": s_info["id"], "class": c_name, "predicted": rles[c_idx]}
                    )

        # Save Submission
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
