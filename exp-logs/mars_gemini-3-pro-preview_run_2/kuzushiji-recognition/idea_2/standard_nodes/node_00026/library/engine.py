import os
import sys
import time
import math
import torch
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# Import from provided library files
from library.utils import calculate_kuzushiji_metrics, format_prediction_string
from library.dataset import KuzushijiDataset, collate_fn


# Set fixed seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


class SmoothedValue:
    """Track a series of values and provide access to smoothed averages over a window or the global series average."""

    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=max(self.deque),
            value=self.deque[-1],
        )


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    metric_logger = defaultdict(SmoothedValue)
    header = f"Epoch: [{epoch}]"

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging purposes (if distributed, but here single GPU)
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger["loss"].update(losses_reduced)
        for k, v in loss_dict_reduced.items():
            metric_logger[k].update(v)

        if i % print_freq == 0:
            print(
                f"{header} Iter: [{i}/{len(data_loader)}] Loss: {metric_logger['loss']}"
            )

    return metric_logger["loss"].global_avg


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # Access dataset to get mappings and GT
    dataset = data_loader.dataset
    int_to_char = dataset.int_to_char

    # Create int -> str image_id map
    int_to_image_id = {v: k for k, v in dataset.image_id_map.items()}

    pred_results = []

    print("Evaluating...")
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for i, (target, output) in enumerate(zip(targets, outputs)):
            img_int_id = target["image_id"].item()
            img_str_id = int_to_image_id.get(img_int_id, str(img_int_id))

            # Post-processing
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            # Rescale boxes to original dimensions
            # Get current (resized) dimensions from the input tensor
            _, h_curr, w_curr = images[i].shape
            w_orig = target["orig_w"].item()
            h_orig = target["orig_h"].item()

            scale_x = w_orig / w_curr
            scale_y = h_orig / h_curr

            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

            label_str = format_prediction_string(boxes, labels, int_to_char)
            pred_results.append({"image_id": img_str_id, "labels": label_str})

    pred_df = pd.DataFrame(pred_results)

    # If no predictions were made (empty dataframe), create empty one with correct columns
    if pred_df.empty:
        pred_df = pd.DataFrame(columns=["image_id", "labels"])

    # Calculate metrics
    # dataset.df contains the Ground Truth
    gt_df = dataset.df

    metrics = calculate_kuzushiji_metrics(gt_df, pred_df)

    print(
        f"Validation Metrics: Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1']}"
    )
    return metrics


def train_kuzushiji_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    device,
    num_epochs=15,
    patience=3,
    save_dir="./working",
):
    os.makedirs(save_dir, exist_ok=True)
    best_f1 = -1.0
    epochs_no_improve = 0
    save_path = os.path.join(save_dir, "best_model.pth")

    # Scheduler: Decay LR by 0.1 at epoch 10 and 13 (assuming 15 epochs total)
    # Using MultiStepLR as per strategy
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 13], gamma=0.1
    )

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        metrics = evaluate(model, val_loader, device)
        val_f1 = metrics["f1"]

        if val_f1 > best_f1:
            print(f"New best F1: {val_f1} (was {best_f1}). Saving model...")
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in F1. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training finished. Best F1: {best_f1}")
    return best_f1


@torch.no_grad()
def inference(model, data_loader, device, output_path="./submission/submission.csv"):
    model.eval()

    dataset = data_loader.dataset
    int_to_char = dataset.int_to_char
    int_to_image_id = {v: k for k, v in dataset.image_id_map.items()}

    results = []

    print("Running inference on test set...")
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for i, (target, output) in enumerate(zip(targets, outputs)):
            img_int_id = target["image_id"].item()
            img_str_id = int_to_image_id.get(img_int_id, str(img_int_id))

            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            # Rescale boxes to original dimensions
            _, h_curr, w_curr = images[i].shape
            w_orig = target["orig_w"].item()
            h_orig = target["orig_h"].item()

            scale_x = w_orig / w_curr
            scale_y = h_orig / h_curr

            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

            label_str = format_prediction_string(boxes, labels, int_to_char)
            results.append({"image_id": img_str_id, "labels": label_str})

    df_sub = pd.DataFrame(results)

    # Ensure all test images are present, even if no predictions
    # Get list of all test IDs from dataset
    all_test_ids = dataset.df["image_id"].unique()

    # Merge with complete list to handle missing predictions (though loop covers all)
    # This ensures order and completeness
    df_template = pd.DataFrame({"image_id": all_test_ids})
    df_final = pd.merge(df_template, df_sub, on="image_id", how="left")
    df_final["labels"] = df_final["labels"].fillna("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
