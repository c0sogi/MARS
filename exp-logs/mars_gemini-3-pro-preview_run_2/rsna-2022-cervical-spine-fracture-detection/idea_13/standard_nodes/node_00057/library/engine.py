import torch
import torch.cuda.amp as amp
import numpy as np
import pandas as pd
import sys
from library.config import Config
from library.utils import competition_metric


def train_one_epoch(model, optimizer, scheduler, dataloader, device, loss_fn, epoch):
    """
    Trains the model for one epoch using Mixed Precision and Gradient Accumulation.
    """
    model.train()

    scaler = amp.GradScaler()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_attn_loss = 0.0
    dataset_size = 0

    optimizer.zero_grad()

    # Iterate over batches
    for step, batch in enumerate(dataloader):
        # Move data to device
        images = batch["images"].to(device, non_blocking=True)  # (B, Seq, 3, H, W)
        targets = batch["targets"].to(device, non_blocking=True)  # (B, 8)
        attn_mask = batch["attn_mask"].to(device, non_blocking=True)  # (B, 7, Seq)
        has_bbox = batch["has_bbox"].to(device, non_blocking=True)  # (B, 1)

        batch_size = images.size(0)

        # Mixed Precision Forward Pass
        with amp.autocast(enabled=True):
            outputs = model(images)

            # Calculate Loss
            # Returns: total_loss, class_loss, attn_loss
            loss, l_cls, l_attn = loss_fn(outputs, targets, attn_mask, has_bbox)

            # Normalize loss for gradient accumulation
            loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward Pass
        scaler.scale(loss).backward()

        # Gradient Accumulation Step
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler Step (if step-based)
            if scheduler is not None:
                scheduler.step()

        # Update Metrics (multiply back by accum steps to get real loss value for logging)
        loss_val = loss.item() * Config.GRAD_ACCUM_STEPS
        running_total_loss += loss_val * batch_size
        running_cls_loss += l_cls.item() * batch_size
        running_attn_loss += l_attn.item() * batch_size
        dataset_size += batch_size

    # Calculate epoch averages
    epoch_loss = running_total_loss / dataset_size
    epoch_cls_loss = running_cls_loss / dataset_size
    epoch_attn_loss = running_attn_loss / dataset_size

    print(
        f"Epoch {epoch} Train | Total Loss: {epoch_loss:.6f} | Cls Loss: {epoch_cls_loss:.6f} | Attn Loss: {epoch_attn_loss:.6f}"
    )

    return epoch_loss


def validate(model, dataloader, device, loss_fn):
    """
    Evaluates the model on the validation set and computes the competition metric.
    """
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    # Containers for metric calculation
    all_study_ids = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)
            has_bbox = batch["has_bbox"].to(device, non_blocking=True)
            study_ids = batch["row_id"]  # List of strings

            batch_size = images.size(0)

            # Forward pass (Mixed Precision usually not strictly needed for inference but saves memory)
            with amp.autocast(enabled=True):
                outputs = model(images)
                loss, _, _ = loss_fn(outputs, targets, attn_mask, has_bbox)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Get probabilities (Sigmoid applied to logits)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets_np)
            all_study_ids.extend(study_ids)

    # Aggregate results
    val_loss = running_loss / dataset_size
    all_probs = np.concatenate(all_probs, axis=0)  # (N_samples, 8)
    all_targets = np.concatenate(all_targets, axis=0)  # (N_samples, 8)

    # --- Format for Competition Metric ---
    # The metric expects a long-format DataFrame with 'row_id' and 'fractured'
    # Columns map: 0->C1, 1->C2, ... 6->C7, 7->patient_overall
    class_names = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    pred_rows = []
    true_rows = []

    for i, uid in enumerate(all_study_ids):
        for class_idx, class_name in enumerate(class_names):
            row_id = f"{uid}_{class_name}"
            prob = all_probs[i, class_idx]
            label = all_targets[i, class_idx]

            pred_rows.append({"row_id": row_id, "fractured": prob})
            true_rows.append({"row_id": row_id, "fractured": label})

    df_pred = pd.DataFrame(pred_rows)
    df_true = pd.DataFrame(true_rows)

    # Calculate Metric
    metric_score = competition_metric(df_true, df_pred)

    print(f"Validation | Loss: {val_loss:.10f} | Metric: {metric_score:.10f}")

    return val_loss, metric_score, df_pred
