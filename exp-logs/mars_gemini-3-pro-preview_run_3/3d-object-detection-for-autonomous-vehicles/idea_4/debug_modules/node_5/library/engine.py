import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.dataset import LyftDataset, collate_fn
from library.model import PointPillars, AnchorGenerator
from library.loss import PointPillarsLoss
from library.utils import setup_logger, decode_boxes, nms_3d


class Trainer:
    def __init__(
        self,
        model_save_path=Config.MODEL_SAVE_PATH,
        load_cached_data=True,
        device=None,
    ):
        """
        Initialize the Trainer.

        Args:
            model_save_path (str): Path to save the best model checkpoint.
            load_cached_data (bool): Whether to load cached dataset files (GT Database).
            device (torch.device): Device to run training on.
        """
        Config.set_seed()
        self.logger = setup_logger(os.path.join(Config.WORKING_DIR, "train_engine.log"))
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_save_path = model_save_path
        self.load_cached_data = load_cached_data

        # Initialize Model components
        self.logger.info("Initializing Model...")
        self.model = PointPillars().to(self.device)

        # Initialize Loss components
        self.anchor_generator = AnchorGenerator()
        self.criterion = PointPillarsLoss(self.anchor_generator).to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.logger.info(f"Trainer initialized on device: {self.device}")

    def fit(
        self,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        patience=3,
        num_workers=Config.NUM_WORKERS,
    ):
        """
        Run the training loop with validation and early stopping.

        Args:
            epochs (int): Maximum number of epochs.
            batch_size (int): Batch size for DataLoaders.
            patience (int): Epochs to wait for improvement before stopping.
            num_workers (int): Number of worker threads for DataLoaders.
        """
        self.logger.info("Loading Datasets...")
        train_ds = LyftDataset(
            Config.TRAIN_METADATA_PATH,
            mode="train",
            load_cached_data=self.load_cached_data,
        )
        val_ds = LyftDataset(
            Config.VAL_METADATA_PATH, mode="val", load_cached_data=self.load_cached_data
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        self.logger.info("Starting Training...")

        for epoch in range(epochs):
            start_time = time.time()

            # --- Training Phase ---
            self.model.train()
            train_loss_meter = 0.0

            # Using tqdm for progress tracking, but logger for metrics
            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
            )

            for batch in pbar:
                pillars = batch["pillars"].to(self.device)
                coords = batch["pillar_coords"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                gt_boxes = batch["gt_boxes"]
                gt_classes = batch["gt_classes"]

                self.optimizer.zero_grad()

                cls_preds, box_preds, dir_preds = self.model(
                    pillars, coords, num_points
                )

                loss_dict = self.criterion(
                    cls_preds,
                    box_preds,
                    dir_preds,
                    gt_boxes,
                    gt_classes,
                )

                # Weighted sum of losses
                loss = (
                    loss_dict["cls_loss"]
                    + loss_dict["loc_loss"] * 2.0
                    + loss_dict["dir_loss"] * 0.2
                )

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_NORM_CLIP)
                self.optimizer.step()
                scheduler.step()

                train_loss_meter += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = train_loss_meter / len(train_loader)

            # --- Validation Phase ---
            avg_val_loss = self.validate(val_loader)

            epoch_time = time.time() - start_time

            # Print full precision metrics
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {avg_val_loss}"
            )

            # --- Early Stopping & Checkpointing ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                self.logger.info(
                    f"Validation loss improved. Model saved to {self.model_save_path}"
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

    def validate(self, loader):
        """
        Evaluate the model on the validation set.

        Args:
            loader (DataLoader): Validation data loader.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc="[Val]", leave=False):
                pillars = batch["pillars"].to(self.device)
                coords = batch["pillar_coords"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                gt_boxes = batch["gt_boxes"]
                gt_classes = batch["gt_classes"]

                cls_preds, box_preds, dir_preds = self.model(
                    pillars, coords, num_points
                )

                loss_dict = self.criterion(
                    cls_preds,
                    box_preds,
                    dir_preds,
                    gt_boxes,
                    gt_classes,
                )

                loss = (
                    loss_dict["cls_loss"]
                    + loss_dict["loc_loss"] * 2.0
                    + loss_dict["dir_loss"] * 0.2
                )
                total_loss += loss.item()

        return total_loss / len(loader)

    def predict(
        self, submission_path=Config.SUBMISSION_PATH, batch_size=Config.BATCH_SIZE
    ):
        """
        Generate predictions for the test set and save to CSV.

        Args:
            submission_path (str): Path to save the submission CSV.
            batch_size (int): Batch size for inference.
        """
        self.logger.info("Generating Submission...")

        # Load best model
        if not os.path.exists(self.model_save_path):
            self.logger.warning(
                f"Model checkpoint not found at {self.model_save_path}. Using current model weights."
            )
        else:
            self.logger.info(f"Loading weights from {self.model_save_path}")
            self.model.load_state_dict(
                torch.load(self.model_save_path, map_location=self.device)
            )

        self.model.eval()

        test_ds = LyftDataset(
            Config.TEST_METADATA_PATH,
            mode="test",
            load_cached_data=self.load_cached_data,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        anchors = self.anchor_generator.get_anchors().to(self.device)
        results = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Inference"):
                pillars = batch["pillars"].to(self.device)
                coords = batch["pillar_coords"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                sample_tokens = batch["sample_tokens"]

                cls_preds, box_preds, dir_preds = self.model(
                    pillars, coords, num_points
                )

                # Post-processing per sample in batch
                B = cls_preds.shape[0]
                for b in range(B):
                    # 1. Decode Scores and Labels
                    # Sigmoid for binary classification per class
                    scores = torch.sigmoid(cls_preds[b])
                    # Get max score and corresponding class index
                    max_scores, labels = scores.max(dim=1)

                    # Filter by score threshold
                    mask = max_scores > Config.SCORE_THRESHOLD

                    if not mask.any():
                        results.append({"Id": sample_tokens[b], "PredictionString": ""})
                        continue

                    # 2. Decode Boxes
                    # box_preds[b]: (N_anchors, 7)
                    valid_box_preds = box_preds[b][mask]
                    valid_anchors = anchors[mask]

                    boxes = decode_boxes(valid_box_preds, valid_anchors)
                    valid_scores = max_scores[mask]
                    valid_labels = labels[mask]  # 0-based index

                    # 3. NMS
                    # Move to CPU for NMS
                    boxes_np = boxes.cpu().numpy()
                    scores_np = valid_scores.cpu().numpy()

                    keep_indices = nms_3d(
                        boxes_np,
                        scores_np,
                        threshold=Config.NMS_IOU_THRESHOLD,
                        max_detections=Config.MAX_DETECTIONS,
                    )

                    # 4. Format Prediction String
                    pred_str_parts = []
                    for k in keep_indices:
                        box = boxes_np[k]
                        sc = scores_np[k]
                        # Convert 0-based label index to 1-based ID, then to class name
                        class_name = Config.ID_TO_CLASS[valid_labels[k].item() + 1]

                        # Format: score x y z w l h yaw class
                        # Note: box is [x, y, z, w, l, h, yaw]
                        pred_str_parts.append(
                            f"{sc:.4f} {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} "
                            f"{box[3]:.4f} {box[4]:.4f} {box[5]:.4f} {box[6]:.4f} {class_name}"
                        )

                    results.append(
                        {
                            "Id": sample_tokens[b],
                            "PredictionString": " ".join(pred_str_parts),
                        }
                    )

        # Save to CSV
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        df.to_csv(submission_path, index=False)
        self.logger.info(f"Submission saved to {submission_path}")
