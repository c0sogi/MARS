import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cv2

from library.config import Config, set_seed
from library.utils import get_device, seed_everything, BCEDiceLoss, rle_encoding
from library.model import HybridSegFormerUNet
from library.dataset import InkDataset


class TrainingEngine:
    """
    Manages the training, validation, and submission generation lifecycle.
    """

    def __init__(self):
        self.device = get_device()
        self.model = HybridSegFormerUNet().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = BCEDiceLoss()
        self.best_score = Config.BASELINE_SCORE

        # Initialize Datasets
        # Config.SAMPLE_SIZE can be used to limit dataset size for debugging
        self.train_dataset = InkDataset(
            Config.TRAIN_METADATA_PATH, mode="train", limit_size=Config.SAMPLE_SIZE
        )
        self.val_dataset = InkDataset(
            Config.VAL_METADATA_PATH, mode="val", limit_size=Config.SAMPLE_SIZE
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            count += 1

        return running_loss / max(count, 1)

    def validate(self):
        """
        Evaluates the model on the validation set using the global F0.5 score.
        """
        self.model.eval()
        tp_total = 0
        fp_total = 0
        fn_total = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                preds = torch.sigmoid(outputs)

                # Binarize predictions and targets
                preds_bin = (preds > Config.THRESHOLD).float()
                targets_bin = (masks > Config.THRESHOLD).float()

                # Accumulate stats for global metric calculation
                tp = (preds_bin * targets_bin).sum().item()
                fp = (preds_bin * (1 - targets_bin)).sum().item()
                fn = ((1 - preds_bin) * targets_bin).sum().item()

                tp_total += tp
                fp_total += fp
                fn_total += fn

        # Calculate F0.5 Score
        beta = Config.METRIC_BETA
        beta_sq = beta**2
        epsilon = 1e-7

        numerator = (1 + beta_sq) * tp_total
        denominator = (1 + beta_sq) * tp_total + beta_sq * fn_total + fp_total

        score = numerator / (denominator + epsilon)
        return score

    def save_checkpoint(self, score):
        """
        Saves the model if the validation score improves upon the baseline.
        """
        if score > self.best_score:
            print(
                f"Validation Score {score} improved over baseline {self.best_score}. Saving model..."
            )
            self.best_score = score
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(self.model.state_dict(), path)
        else:
            print(f"Validation Score {score} did not improve over {self.best_score}.")

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training for {Config.EPOCHS} epochs...")
        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_score = self.validate()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val F0.5: {val_score}"
            )
            self.save_checkpoint(val_score)

    def generate_submission(self):
        """
        Generates the submission file using Decoupled Z-Scanning inference.
        """
        print("Generating submission with Decoupled Z-Scanning...")

        # Load best model if available
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model.")
        else:
            print("No best model found. Using current model state.")

        self.model.eval()

        # Initialize datasets for scanning offsets
        dataloaders = []
        for offset in Config.SCAN_OFFSETS:
            z = Config.Z_START + offset
            ds = InkDataset(Config.TEST_METADATA_PATH, mode="test", z_start=z)
            dl = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            dataloaders.append(dl)

        # Prepare storage for reconstructed probability maps
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        fragment_maps = {}
        fragment_shapes = {}

        for _, row in test_df.iterrows():
            fid = row["fragment_id"]
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            h, w = mask.shape
            fragment_shapes[fid] = (h, w)
            fragment_maps[fid] = np.zeros((h, w), dtype=np.float32)

        # Iterate through all dataloaders simultaneously (Max Fusion)
        with torch.no_grad():
            for batches in zip(*dataloaders):
                # batches is a tuple of ( (imgs, coords, fids), ... ) for each offset

                imgs_list = [b[0].to(self.device) for b in batches]
                coords = batches[0][1]  # (B, 2)
                fids = batches[0][2]  # tuple of strings

                # Predict for each offset
                preds_list = []
                for img in imgs_list:
                    out = self.model(img)
                    pred = torch.sigmoid(out)
                    preds_list.append(pred)

                # Max Fusion across the stack dimension
                stack = torch.stack(preds_list, dim=0)  # (Scans, B, 1, H, W)
                fused_pred, _ = torch.max(stack, dim=0)  # (B, 1, H, W)

                fused_pred = fused_pred.cpu().numpy()
                coords = coords.numpy()

                # Reconstruct full maps
                for i, fid in enumerate(fids):
                    x, y = coords[i]
                    p_map = fused_pred[i, 0]  # (512, 512)

                    H, W = fragment_shapes[fid]

                    # Determine crop size (handle boundaries)
                    h_p, w_p = p_map.shape
                    y_end = min(y + h_p, H)
                    x_end = min(x + w_p, W)

                    h_use = y_end - y
                    w_use = x_end - x

                    current_val = fragment_maps[fid][y:y_end, x:x_end]
                    new_val = p_map[:h_use, :w_use]

                    # Update map with max probability
                    fragment_maps[fid][y:y_end, x:x_end] = np.maximum(
                        current_val, new_val
                    )

        # Generate RLE Submission
        submission_data = []
        for fid in sorted(fragment_maps.keys()):
            prob_map = fragment_maps[fid]
            binary_map = prob_map > Config.THRESHOLD

            rle = rle_encoding(binary_map)
            submission_data.append({"Id": fid, "Predicted": rle})

        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
