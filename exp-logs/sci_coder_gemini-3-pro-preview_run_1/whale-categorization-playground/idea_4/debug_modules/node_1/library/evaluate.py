import os
import torch
import pandas as pd
import numpy as np
from library.utils import AverageMeter, map_at_5


def validate(val_loader, model, criterion, device, classes):
    """
    Evaluates the model on the validation set using Test-Time Augmentation (TTA).
    Computes Loss and MAP@5.

    Args:
        val_loader (DataLoader): Validation data loader.
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        classes (list/array): List of class names corresponding to indices.

    Returns:
        float: Average Loss.
        float: MAP@5 Score.
    """
    model.eval()

    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # ---------------------------------------------------------
            # Test-Time Augmentation (TTA): Horizontal Flip
            # ---------------------------------------------------------
            # 1. Forward pass with original images
            # Passing labels=None returns raw scaled cosine similarities (logits)
            logits_orig = model(images, labels=None)

            # 2. Forward pass with flipped images
            # Flip width dimension (B, C, H, W) -> dim 3
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, labels=None)

            # 3. Average logits
            logits = (logits_orig + logits_flip) / 2.0

            # ---------------------------------------------------------
            # Metrics
            # ---------------------------------------------------------
            # Calculate loss using the averaged logits
            loss = criterion(logits, labels)
            losses.update(loss.item(), images.size(0))

            # Get Top 5 predictions
            # logits are (Batch, Num_Classes)
            _, top_indices = torch.topk(logits, k=5, dim=1)

            # Convert indices to class names
            top_indices = top_indices.cpu().numpy()
            labels_np = labels.cpu().numpy()

            batch_preds = []
            for idx_list in top_indices:
                pred_names = [classes[i] for i in idx_list]
                batch_preds.append(pred_names)

            # Convert targets to class names
            batch_targets = [classes[i] for i in labels_np]

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)

    # Calculate MAP@5
    map5_score = map_at_5(all_preds, all_targets)

    return losses.avg, map5_score


def inference(test_loader, model, device, classes):
    """
    Generates predictions for the test set using TTA and saves to submission file.

    Args:
        test_loader (DataLoader): Test data loader.
        model (nn.Module): The trained model.
        device (torch.device): Device to run inference on.
        classes (list/array): List of class names corresponding to indices.
    """
    model.eval()

    image_ids = []
    predictions = []

    print("Generating submission predictions with TTA...")

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            # TTA: Horizontal Flip
            logits_orig = model(images, labels=None)

            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, labels=None)

            logits = (logits_orig + logits_flip) / 2.0

            # Get Top 5
            _, top_indices = torch.topk(logits, k=5, dim=1)
            top_indices = top_indices.cpu().numpy()

            for i, filename in enumerate(filenames):
                image_ids.append(filename)

                # Map indices to class names
                pred_labels = [classes[idx] for idx in top_indices[i]]
                # Format: "label1 label2 label3 label4 label5"
                predictions.append(" ".join(pred_labels))

    # Create DataFrame
    df_sub = pd.DataFrame({"Image": image_ids, "Id": predictions})

    # Save
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
