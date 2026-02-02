import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from library.utils import AverageMeter


def train_one_epoch(model, loader, optimizer, device, epoch, scheduler=None):
    """
    Trains the model for one epoch using Weighted Multi-Task Cross-Entropy Loss.

    Loss Formula: L_total = L_species + 0.2 * L_genus + 0.1 * L_family
    """
    model.train()

    # Loss trackers
    losses = AverageMeter()
    losses_species = AverageMeter()
    losses_genus = AverageMeter()
    losses_family = AverageMeter()

    # Criterion with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    for step, batch in enumerate(loader):
        # Unpack batch: (images, species_label, genus_label, family_label)
        images = batch[0].to(device, non_blocking=True)
        species_targets = batch[1].to(device, non_blocking=True)
        genus_targets = batch[2].to(device, non_blocking=True)
        family_targets = batch[3].to(device, non_blocking=True)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass (Cascading Model returns 3 logits)
        species_logits, genus_logits, family_logits = model(images)

        # Compute losses
        loss_s = criterion(species_logits, species_targets)
        loss_g = criterion(genus_logits, genus_targets)
        loss_f = criterion(family_logits, family_targets)

        # Weighted sum
        loss = loss_s + 0.2 * loss_g + 0.1 * loss_f

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step scheduler (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        # Update trackers
        losses.update(loss.item(), batch_size)
        losses_species.update(loss_s.item(), batch_size)
        losses_genus.update(loss_g.item(), batch_size)
        losses_family.update(loss_f.item(), batch_size)

    print(f"Epoch {epoch} Training Results:")
    print(f"  Avg Total Loss: {losses.avg}")
    print(f"  Avg Species Loss: {losses_species.avg}")
    print(f"  Avg Genus Loss: {losses_genus.avg}")
    print(f"  Avg Family Loss: {losses_family.avg}")

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Metric: Macro F1 Score on Species.
    """
    model.eval()

    all_preds = []
    all_targets = []
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            species_targets = batch[1].to(device, non_blocking=True)
            genus_targets = batch[2].to(device, non_blocking=True)
            family_targets = batch[3].to(device, non_blocking=True)

            # Forward pass
            species_logits, genus_logits, family_logits = model(images)

            # Calculate validation loss for monitoring
            loss_s = criterion(species_logits, species_targets)
            loss_g = criterion(genus_logits, genus_targets)
            loss_f = criterion(family_logits, family_targets)
            loss = loss_s + 0.2 * loss_g + 0.1 * loss_f
            losses.update(loss.item(), images.size(0))

            # Get predictions for species
            preds = torch.argmax(species_logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(species_targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Macro F1
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    print(f"Validation Results:")
    print(f"  Loss: {losses.avg}")
    print(f"  Macro F1 Score: {macro_f1}")

    return macro_f1, losses.avg


def predict_tta(model, loader, device):
    """
    Performs inference using Horizontal Flip Test Time Augmentation (TTA).
    Averages probabilities from original and flipped images.

    Returns:
        pd.DataFrame: DataFrame with 'Id' and 'Predicted' columns.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            # Test loader returns (image, image_id)
            images = batch[0].to(device, non_blocking=True)
            image_ids = batch[1]

            # 1. Forward pass original
            logits_1, _, _ = model(images)
            probs_1 = torch.softmax(logits_1, dim=1)

            # 2. Forward pass flipped (Horizontal Flip)
            # Assuming images are (B, C, H, W), dim=3 is width
            images_flip = torch.flip(images, dims=[3])
            logits_2, _, _ = model(images_flip)
            probs_2 = torch.softmax(logits_2, dim=1)

            # 3. Average probabilities
            avg_probs = (probs_1 + probs_2) / 2.0

            # 4. Get predictions
            batch_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

            ids_list.extend(image_ids)
            preds_list.extend(batch_preds)

    # Create DataFrame
    submission_df = pd.DataFrame({"Id": ids_list, "Predicted": preds_list})

    return submission_df


def generate_submission(model, loader, device, save_path="./submission/submission.csv"):
    """
    Generates predictions and saves them to a CSV file.
    """
    print("Starting TTA Inference...")
    df = predict_tta(model, loader, device)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {df.shape}")
    print(df.head())
