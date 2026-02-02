import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel

from library.utils import MetricMonitor
from library.dataset import mixup_data


def update_bn_custom(loader, model, device):
    """
    Custom Batch Normalization update for SWA that handles the specific
    return format of the CactusDataset (image, label, quality).
    """
    model.train()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                # Dataset returns (img, label, qual), we only need img
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            model(x)


def train_one_epoch(model, train_loader, optimizer, device, epoch, config):
    """
    Trains the model for one epoch using Mixup and Multi-Task Loss.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Configuration
    alpha = config.get("mixup_alpha", 0.2)
    qual_weight = config.get("quality_weight", 1.0)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_qual = nn.MSELoss()

    for batch_idx, (images, labels, qualities) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        qualities = qualities.to(device)

        # Apply Mixup
        mixed_images, labels_a, labels_b, qual_a, qual_b, lam = mixup_data(
            images, labels, qualities, alpha=alpha, device=device
        )

        optimizer.zero_grad()

        # Forward pass (Multi-head)
        cls_preds, qual_preds = model(mixed_images)

        # Reshape targets to (B, 1) to match model output
        labels_a = labels_a.view(-1, 1)
        labels_b = labels_b.view(-1, 1)
        qual_a = qual_a.view(-1, 1)
        qual_b = qual_b.view(-1, 1)

        # Compute Weighted Multi-Task Loss
        loss_cls = lam * criterion_cls(cls_preds, labels_a) + (1 - lam) * criterion_cls(
            cls_preds, labels_b
        )
        loss_qual = lam * criterion_qual(qual_preds, qual_a) + (
            1 - lam
        ) * criterion_qual(qual_preds, qual_b)

        loss = loss_cls + (qual_weight * loss_qual)

        # Backward and Optimize
        loss.backward()
        optimizer.step()

        # Update Metrics
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Loss_Cls", loss_cls.item())
        metric_monitor.update("Loss_Qual", loss_qual.item())

    return metric_monitor.get_metrics()


def validate(model, val_loader, device, config):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    qual_weight = config.get("quality_weight", 1.0)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_qual = nn.MSELoss()

    all_cls_preds = []
    all_cls_targets = []

    with torch.no_grad():
        for images, labels, qualities in val_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            qualities = qualities.to(device).view(-1, 1)

            cls_preds, qual_preds = model(images)

            loss_cls = criterion_cls(cls_preds, labels)
            loss_qual = criterion_qual(qual_preds, qualities)
            loss = loss_cls + (qual_weight * loss_qual)

            metric_monitor.update("Val_Loss", loss.item())
            metric_monitor.update("Val_Loss_Cls", loss_cls.item())
            metric_monitor.update("Val_Loss_Qual", loss_qual.item())

            # Store predictions for AUC calculation
            all_cls_preds.append(torch.sigmoid(cls_preds).cpu().numpy())
            all_cls_targets.append(labels.cpu().numpy())

    # Compute ROC AUC
    all_cls_preds = np.concatenate(all_cls_preds)
    all_cls_targets = np.concatenate(all_cls_targets)

    try:
        auc = roc_auc_score(all_cls_targets, all_cls_preds)
    except ValueError:
        auc = 0.5

    metric_monitor.update("AUC", auc)

    return metric_monitor.get_metrics()


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config):
    """
    Orchestrates the training process including SWA and Early Stopping.
    """
    epochs = config.get("epochs", 30)
    swa_start = config.get("swa_start_epoch", 20)
    patience = config.get("patience", 7)
    save_dir = config.get("save_dir", "./working/idea_26")

    os.makedirs(save_dir, exist_ok=True)

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # SWA Initialization
    swa_model = AveragedModel(model)

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config
        )

        # Validate
        val_metrics = validate(model, val_loader, device, config)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # SWA Update
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        # Logging (Full precision)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train: {train_metrics}")
        print(f"Val  : {val_metrics}")

        # Early Stopping & Checkpointing (Monitoring Standard Model)
        current_auc = val_metrics["AUC"]
        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Stop early if patience exceeded and we are not in SWA phase (or SWA hasn't started)
        if patience_counter >= patience and epoch < swa_start:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Finalize SWA if applicable
    if epochs >= swa_start:
        print("Updating SWA Batch Norm statistics...")
        update_bn_custom(train_loader, swa_model, device=device)

        swa_path = os.path.join(save_dir, "swa_model.pth")
        torch.save(swa_model.state_dict(), swa_path)

        # Validate SWA Model
        swa_metrics = validate(swa_model, val_loader, device, config)
        print(f"SWA Val Metrics: {swa_metrics}")

        return swa_model, swa_metrics

    # Load and return best model if SWA was not used
    print("Loading best model from checkpoint...")
    model.load_state_dict(torch.load(best_model_path))
    return model, {"AUC": best_auc}


def predict_test_set(model, test_loader, device):
    """
    Generates predictions using 4-view Test Time Augmentation (TTA).
    Returns averaged probabilities and quality predictions.
    """
    model.eval()
    preds = []
    qual_preds = []

    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            bs = images.size(0)

            # TTA: Original, HFlip, VFlip, HVFlip
            x1 = images
            x2 = torch.flip(images, [3])
            x3 = torch.flip(images, [2])
            x4 = torch.flip(images, [2, 3])

            # Stack for batch processing
            x_tta = torch.cat([x1, x2, x3, x4], dim=0)

            # Forward
            logits, q_out = model(x_tta)
            probs = torch.sigmoid(logits)

            # Split and Average
            p1, p2, p3, p4 = torch.split(probs, bs)
            q1, q2, q3, q4 = torch.split(q_out, bs)

            avg_prob = (p1 + p2 + p3 + p4) / 4.0
            avg_q = (q1 + q2 + q3 + q4) / 4.0

            preds.append(avg_prob.cpu().numpy())
            qual_preds.append(avg_q.cpu().numpy())

    return np.concatenate(preds), np.concatenate(qual_preds)


def generate_submission(
    model, test_loader, device, output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    print("Generating submission with TTA...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Predict
    probs, _ = predict_test_set(model, test_loader, device)

    # Retrieve IDs
    # Assuming test_loader.dataset is the CactusDataset which has .ids attribute
    if hasattr(test_loader.dataset, "ids"):
        ids = test_loader.dataset.ids
    else:
        # Fallback if dataset is wrapped (e.g. Subset), though not expected here
        raise AttributeError("Dataset does not have 'ids' attribute.")

    # Create DataFrame and Save
    df = pd.DataFrame({"id": ids, "has_cactus": probs.flatten()})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
