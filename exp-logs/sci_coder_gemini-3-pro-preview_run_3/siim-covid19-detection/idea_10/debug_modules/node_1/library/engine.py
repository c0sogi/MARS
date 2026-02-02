import math
import sys
import time
import os
import torch
import pandas as pd
import numpy as np
from typing import Iterable, Dict, List

from library.config import Config
from library.utils import collate_fn
from library.model import build_model
from library.loss import build_criterion
from library.dataset import ChestXrayDataset
from torch.utils.data import DataLoader


class Engine:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = 3  # Early stopping patience

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        max_norm: float = 0,
    ):
        model.train()
        criterion.train()

        total_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for samples, targets, _ in data_loader:
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

            # Weighted sum of losses is handled inside criterion via weight_dict,
            # but criterion returns individual components. We need to sum them based on weights.
            # However, looking at library/loss.py, build_criterion sets weights in weight_dict.
            # The CoDETRLoss returns a dictionary of losses. We need to apply weights here or
            # assume the loss function returns raw losses and we weight them.
            # The provided loss.py implementation calculates raw losses.
            # We must apply the weights from criterion.weight_dict.

            losses = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict.keys()
                if k in criterion.weight_dict
            )

            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            total_loss += loss_value
            num_batches += 1

        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f} | Time: {elapsed:.0f}s")
        return avg_loss

    @torch.no_grad()
    def evaluate(
        self, model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable
    ):
        model.eval()
        criterion.eval()

        total_loss = 0.0
        num_batches = 0

        for samples, targets, _ in data_loader:
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

            losses = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict.keys()
                if k in criterion.weight_dict
            )
            total_loss += losses.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def get_optimizer(self, model):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": Config.LR,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": Config.BACKBONE_LR,
            },
        ]
        return torch.optim.AdamW(
            param_dicts, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

    def run_training(self):
        # 1. Load Data
        print("Loading datasets...")
        train_dataset = ChestXrayDataset(split="train", load_cached_data=True)
        val_dataset = ChestXrayDataset(split="val", load_cached_data=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=Config.PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=Config.PIN_MEMORY,
        )

        # 2. Build Model & Criterion
        print("Building model...")
        model = build_model(Config)
        model.to(self.device)
        criterion = build_criterion(Config)
        criterion.to(self.device)

        optimizer = self.get_optimizer(model)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, Config.LR_DROP)

        print(f"Start training for {Config.EPOCHS} epochs...")
        for epoch in range(1, Config.EPOCHS + 1):
            self.train_one_epoch(
                model, criterion, train_loader, optimizer, epoch, Config.CLIP_MAX_NORM
            )
            lr_scheduler.step()

            val_loss = self.evaluate(model, criterion, val_loader)

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved with loss {val_loss:.6f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

    @torch.no_grad()
    def run_inference(self):
        print("Running inference on test set...")

        # Load Test Data
        test_dataset = ChestXrayDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=Config.PIN_MEMORY,
        )

        # Load Best Model
        model = build_model(Config)
        if os.path.exists(Config.BEST_MODEL_PATH):
            model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model.")
        else:
            print("Warning: Best model not found, using initialized weights.")

        model.to(self.device)
        model.eval()

        results = []

        # Mapping for study aggregation
        # We need to map image_id to study_id to aggregate study predictions
        # The test_dataset.df has this info
        test_df = test_dataset.df
        image_to_study = dict(zip(test_df.image_id, test_df.study_id))

        study_predictions = {}  # study_id -> list of prob vectors

        for samples, targets, image_ids in test_loader:
            samples = samples.to(self.device)
            # targets contains 'orig_size' which is [H, W]

            outputs = model(samples)

            # Process Batch
            pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
            pred_boxes = outputs["pred_boxes"]  # [B, Q, 4] (cx, cy, w, h) normalized
            pred_study = outputs["pred_study"]  # [B, 4]

            probs = pred_logits.sigmoid()
            study_probs = pred_study.softmax(dim=-1)

            for i, img_id in enumerate(image_ids):
                # 1. Study Level Aggregation
                study_id = image_to_study.get(img_id)
                if study_id:
                    if study_id not in study_predictions:
                        study_predictions[study_id] = []
                    study_predictions[study_id].append(study_probs[i].cpu().numpy())

                # 2. Image Level Prediction
                orig_h, orig_w = targets[i]["orig_size"]

                # Check study prediction for this specific image context (logic from idea)
                # We use the study prediction for this image to gate the boxes
                # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
                curr_study_pred_idx = study_probs[i].argmax().item()

                prediction_string = "none 1 0 0 1 1"

                # If NOT Negative for Pneumonia (Index 0), look for boxes
                if curr_study_pred_idx != 0:
                    # Filter boxes
                    # Class 0 is opacity. Config.NUM_CLASSES=1.
                    # probs shape [Q, 2] (opacity, bg) or [Q, 1] depending on implementation.
                    # Model outputs num_classes+1 usually.
                    # We take score for class 0.
                    scores = probs[i, :, 0]
                    keep = scores > Config.CONF_THRESHOLD

                    if keep.any():
                        valid_scores = scores[keep]
                        valid_boxes = pred_boxes[i, keep]  # [N, 4] normalized

                        # Convert to pixel coords [xmin, ymin, xmax, ymax]
                        cx, cy, w, h = valid_boxes.unbind(1)
                        cx = cx * orig_w
                        cy = cy * orig_h
                        w = w * orig_w
                        h = h * orig_h

                        xmin = cx - w / 2
                        ymin = cy - h / 2
                        xmax = cx + w / 2
                        ymax = cy + h / 2

                        # Format string
                        box_strs = []
                        for s, x1, y1, x2, y2 in zip(
                            valid_scores, xmin, ymin, xmax, ymax
                        ):
                            box_strs.append(
                                f"opacity {s.item():.4f} {x1.item():.4f} {y1.item():.4f} {x2.item():.4f} {y2.item():.4f}"
                            )

                        prediction_string = " ".join(box_strs)

                results.append(
                    {"id": f"{img_id}_image", "PredictionString": prediction_string}
                )

        # Process Study Level Predictions
        # Average probabilities per study
        study_labels = ["negative", "typical", "indeterminate", "atypical"]

        for study_id, prob_list in study_predictions.items():
            avg_probs = np.mean(prob_list, axis=0)
            best_idx = np.argmax(avg_probs)
            confidence = avg_probs[best_idx]
            label_name = study_labels[best_idx]

            results.append(
                {
                    "id": f"{study_id}_study",
                    "PredictionString": f"{label_name} {confidence:.4f} 0 0 1 1",
                }
            )

        # Save Submission
        submission_df = pd.DataFrame(results)
        # Ensure correct column order
        submission_df = submission_df[["id", "PredictionString"]]
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    engine = Engine()
    engine.run_training()
    engine.run_inference()


if __name__ == "__main__":
    main()
