import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import TwoStagePointPillars
from library.dataset import LidarDataset
from library.utils import iou3d_shapely

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def parse_prediction_string(pred_str):
    """
    Parses a prediction string into a list of dictionaries.
    Format: score x y z w l h yaw class
    """
    if not isinstance(pred_str, str) or not pred_str.strip():
        return []

    parts = pred_str.strip().split()
    stride = 9
    num_preds = len(parts) // stride
    preds = []

    for i in range(num_preds):
        offset = i * stride
        try:
            score = float(parts[offset])
            box = [
                float(parts[offset + j]) for j in range(1, 8)
            ]  # x, y, z, w, l, h, yaw
            class_name = parts[offset + 8]
            preds.append(
                {
                    "box": np.array(box, dtype=np.float32),
                    "score": score,
                    "class_name": class_name,
                }
            )
        except ValueError:
            continue
    return preds


def compute_matches(
    gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, iou_threshold
):
    """
    Matches predictions to GT boxes greedily based on confidence and IoU.
    Returns TP, FP, FN counts.
    """
    # Sort predictions by score descending
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_classes = pred_classes[sorted_indices]

    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_gt == 0:
        return 0, num_pred, 0  # TP, FP, FN

    if num_pred == 0:
        return 0, 0, num_gt

    # Calculate IoU matrix (N_pred x N_gt)
    iou_matrix = iou3d_shapely(pred_boxes, gt_boxes)

    gt_matched = np.zeros(num_gt, dtype=bool)
    pred_matched = np.zeros(num_pred, dtype=bool)

    tp = 0

    for p_idx in range(num_pred):
        if pred_matched[p_idx]:
            continue

        # Find best match in GT
        best_iou = -1.0
        best_gt_idx = -1

        # Get IoUs for this prediction
        ious = iou_matrix[p_idx]

        # Filter by class and threshold
        # We only consider GTs that have the same class
        p_cls = pred_classes[p_idx]

        # Optimization: Check potential matches
        candidates = np.where((ious > iou_threshold) & (~gt_matched))[0]

        for gt_idx in candidates:
            if gt_classes[gt_idx] == p_cls:
                if ious[gt_idx] > best_iou:
                    best_iou = ious[gt_idx]
                    best_gt_idx = gt_idx

        if best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = True
            pred_matched[p_idx] = True

    fp = num_pred - tp
    fn = num_gt - tp

    return tp, fp, fn


def calculate_metric_score(gt_data, pred_str):
    """
    Calculates the specific metric for one image.
    Metric: Mean over thresholds (0.5 to 0.95) of TP / (TP + FP + FN).
    """
    # Parse Predictions
    preds = parse_prediction_string(pred_str)

    if not preds:
        pred_boxes = np.zeros((0, 7), dtype=np.float32)
        pred_classes = np.array([])
        pred_scores = np.array([])
    else:
        pred_boxes = np.array([p["box"] for p in preds])
        pred_classes = np.array([p["class_name"] for p in preds])
        pred_scores = np.array([p["score"] for p in preds])

    # Parse GT
    # gt_data is list of (box, class_name) tuples
    if len(gt_data) == 0:
        gt_boxes = np.zeros((0, 7), dtype=np.float32)
        gt_classes = np.array([])
    else:
        gt_boxes = np.array([g[0] for g in gt_data])
        gt_classes = np.array([g[1] for g in gt_data])

    # Handle Empty Cases
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0  # Perfect empty prediction
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return 0.0  # False positives with no GT

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = []

    for t in thresholds:
        tp, fp, fn = compute_matches(
            gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, t
        )

        denominator = tp + fp + fn
        if denominator == 0:
            prec = 1.0
        else:
            prec = tp / denominator
        precisions.append(prec)

    return np.mean(precisions)


# -----------------------------------------------------------------------------
# Engine Class
# -----------------------------------------------------------------------------


class Engine:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.model = TwoStagePointPillars().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Placeholder for scheduler (initialized in train)
        self.scheduler = None

    def train_one_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0.0
        total_s1_loss = 0.0
        total_s2_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            points = [p.to(self.device) for p in batch["points"]]
            gt_boxes = [b.to(self.device) for b in batch["gt_boxes"]]
            gt_labels = [l.to(self.device) for l in batch["gt_labels"]]

            self.optimizer.zero_grad()

            loss, stats = self.model(points, gt_boxes, gt_labels)

            if torch.isnan(loss):
                continue

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRAD_NORM_CLIP
            )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            total_s1_loss += stats.get("stage1_loss", 0.0)
            total_s2_loss += stats.get("stage2_loss", 0.0)
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_s1 = total_s1_loss / max(1, num_batches)
        avg_s2 = total_s2_loss / max(1, num_batches)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch_idx+1} | Loss: {avg_loss:.6f} (S1: {avg_s1:.6f}, S2: {avg_s2:.6f}) | Time: {elapsed:.2f}s"
        )

        return avg_loss

    def evaluate(self, dataloader):
        self.model.eval()
        scores = []

        print("Evaluating...")

        with torch.no_grad():
            for batch in dataloader:
                points = [p.to(self.device) for p in batch["points"]]
                metadata = batch["metadata"]

                # Inference
                results = self.model(points)

                # Process batch
                for i, meta in enumerate(metadata):
                    # Get Prediction String
                    pred_str = results.get(i, "")

                    # Get GT Data
                    gt_b = batch["gt_boxes"][i].cpu().numpy()
                    gt_l = batch["gt_labels"][i].cpu().numpy()

                    gt_data = []
                    for k in range(len(gt_b)):
                        cls_idx = gt_l[k]
                        cls_name = Config.CLASS_NAMES[cls_idx]
                        gt_data.append((gt_b[k], cls_name))

                    # Calculate Score
                    score = calculate_metric_score(gt_data, pred_str)
                    scores.append(score)

        mean_score = np.mean(scores) if scores else 0.0
        print(f"Validation Metric (mAP @ IoU 0.5:0.95): {mean_score:.10f}")
        return mean_score

    def train(self, train_loader, val_loader, epochs=Config.NUM_EPOCHS):
        # Scheduler: OneCycleLR
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100,
        )

        best_metric = 0.0

        for epoch in range(epochs):
            self.train_one_epoch(train_loader, epoch)

            # Validate every epoch
            metric = self.evaluate(val_loader)

            # Save Checkpoint
            if metric > best_metric:
                best_metric = metric
                save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with score: {best_metric:.6f}")

            # Regular checkpoint
            torch.save(
                self.model.state_dict(),
                os.path.join(Config.WORKING_DIR, "last_model.pth"),
            )

    def generate_submission(self, test_loader, output_path=Config.SUBMISSION_PATH):
        self.model.eval()

        # Load best model if available
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for submission...")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
        else:
            print("Warning: Best model not found, using current weights.")

        ids = []
        pred_strings = []

        print("Generating submission...")
        with torch.no_grad():
            for batch in test_loader:
                points = [p.to(self.device) for p in batch["points"]]
                metadata = batch["metadata"]

                results = self.model(points)

                for i, meta in enumerate(metadata):
                    sample_token = meta["sample_token"]
                    pred_str = results.get(i, "")

                    ids.append(sample_token)
                    pred_strings.append(pred_str)

        df = pd.DataFrame({"Id": ids, "PredictionString": pred_strings})
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


# -----------------------------------------------------------------------------
# Main Execution Function
# -----------------------------------------------------------------------------


def run_engine(epochs=None):
    # Override epochs if provided
    num_epochs = epochs if epochs is not None else Config.NUM_EPOCHS

    # 1. Datasets
    print("Initializing Datasets...")
    train_dataset = LidarDataset(split="train")
    val_dataset = LidarDataset(split="val")
    test_dataset = LidarDataset(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=LidarDataset.collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=LidarDataset.collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=LidarDataset.collate_fn,
        pin_memory=True,
    )

    # 2. Engine
    engine = Engine()

    # 3. Train
    print(f"Starting Training for {num_epochs} epochs...")
    engine.train(train_loader, val_loader, epochs=num_epochs)

    # 4. Submit
    engine.generate_submission(test_loader)
