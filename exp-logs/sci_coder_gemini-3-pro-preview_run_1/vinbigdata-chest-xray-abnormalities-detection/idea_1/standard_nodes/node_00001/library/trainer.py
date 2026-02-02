import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random

from library.config import Config
from library.dataset import VinBigDataDataset
from library.model import CenterNet
from library.loss import CenterNetLoss
from library.inference import (
    decode_predictions,
    rescale_bboxes,
    convert_to_prediction_string,
)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, load_cached_data=False):
        self.device = torch.device(Config.DEVICE)
        self.load_cached_data = load_cached_data

        # Reproducibility
        seed_everything(Config.SEED)

        # Directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        self.model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        # Data Loaders
        self.train_dataset = VinBigDataDataset(
            split="train", debug=Config.DEBUG, load_cached_data=load_cached_data
        )
        self.val_dataset = VinBigDataDataset(
            split="val", debug=Config.DEBUG, load_cached_data=load_cached_data
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

        # Model, Loss, Optimizer
        self.model = CenterNet(pretrained=True)
        self.model.to(self.device)

        self.criterion = CenterNetLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Optional but recommended)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_hm_loss = 0.0
        running_wh_loss = 0.0
        running_reg_loss = 0.0

        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Move data to device
            imgs = batch_data["image"].to(self.device)

            # Forward pass
            outputs = self.model(imgs)

            # Calculate loss
            loss, loss_stats = self.criterion(outputs, batch_data)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            running_hm_loss += loss_stats["hm_loss"].item()
            running_wh_loss += loss_stats["wh_loss"].item()
            running_reg_loss += loss_stats["reg_loss"].item()

        avg_loss = running_loss / num_batches
        print(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Train Loss: {avg_loss:.6f} "
            f"(HM: {running_hm_loss/num_batches:.6f}, "
            f"WH: {running_wh_loss/num_batches:.6f}, "
            f"Reg: {running_reg_loss/num_batches:.6f})"
        )

        return avg_loss

    def evaluate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                imgs = batch_data["image"].to(self.device)

                outputs = self.model(imgs)
                loss, _ = self.criterion(outputs, batch_data)

                running_loss += loss.item()

        avg_loss = running_loss / num_batches
        # Print full precision as requested
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Validation Loss: {avg_loss}")

        return avg_loss

    def train(self):
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch)

            self.scheduler.step(val_loss)

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"  New best model saved with loss: {best_val_loss}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print("Training completed.")

    def predict(self):
        print("Starting inference on test set...")

        # Load Best Model
        if os.path.exists(self.model_save_path):
            self.model.load_state_dict(
                torch.load(self.model_save_path, map_location=self.device)
            )
            print(f"Loaded model from {self.model_save_path}")
        else:
            print("Warning: No saved model found. Using current model weights.")

        self.model.eval()

        # Test Dataset
        test_dataset = VinBigDataDataset(
            split="test", debug=Config.DEBUG, load_cached_data=self.load_cached_data
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        results = []

        # Get original image dimensions for rescaling
        # We need to access the dataframe from the dataset to get original sizes if they were available,
        # but the dataset loads images on the fly.
        # However, rescale_bboxes expects (height, width).
        # Since we don't have easy access to original DICOM headers here without reloading,
        # we will rely on the fact that VinBigDataDataset loads images.
        # Note: The provided dataset class does not return original shape in __getitem__ for test set easily
        # unless we modify it.
        # However, looking at library/dataset.py, __getitem__ loads the image.
        # We can infer the original size if we had it.
        # Wait, the provided `rescale_bboxes` function requires `original_shapes`.
        # The `VinBigDataDataset` resizes images to `Config.IMG_SIZE`.
        # We need the original dimensions.
        # Strategy: We will read the original image dimensions during the inference loop.
        # Since `VinBigDataDataset` loads the image to create the tensor, but returns the resized tensor,
        # we might lose the original info.
        # BUT, `VinBigDataDataset` has `self.df`. We can look up the file path.
        # Ideally, the dataset should return `original_shape`.
        # Since I cannot modify `dataset.py`, I will retrieve original dimensions manually using the image_id.

        # Pre-fetch metadata to get paths
        test_meta = pd.read_csv(Config.TEST_META_PATH)
        path_map = dict(zip(test_meta["image_id"], test_meta["file_path"]))

        import cv2

        try:
            import pydicom
        except ImportError:
            pydicom = None

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                imgs = batch_data["image"].to(self.device)
                image_ids = batch_data["image_id"]

                # Forward
                outputs = self.model(imgs)

                # Decode
                detections = decode_predictions(
                    outputs["hm"], outputs["wh"], outputs["reg"], K=Config.TOP_K
                )

                # Get original shapes for this batch
                original_shapes = []
                for img_id in image_ids:
                    # We need to find the original size.
                    # This is expensive but necessary since dataset doesn't return it.
                    f_path = path_map.get(img_id)
                    full_path = os.path.join(Config.INPUT_DIR, f_path)

                    h, w = Config.IMG_SIZE, Config.IMG_SIZE  # Default fallback

                    # Try reading header only for speed
                    if pydicom:
                        try:
                            dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
                            h, w = dcm.Rows, dcm.Columns
                        except:
                            pass

                    # Fallback to CV2 if pydicom failed or not present, but CV2 reads pixels
                    if (h == Config.IMG_SIZE) and os.path.exists(full_path):
                        try:
                            # Just read shape
                            img_tmp = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                            if img_tmp is not None:
                                h, w = img_tmp.shape[:2]
                        except:
                            pass

                    original_shapes.append((h, w))

                # Rescale
                rescaled_dets = rescale_bboxes(detections, original_shapes)

                # Format strings
                for i, img_id in enumerate(image_ids):
                    pred_str = convert_to_prediction_string(rescaled_dets[i])
                    results.append({"image_id": img_id, "PredictionString": pred_str})

        # Save Submission
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
