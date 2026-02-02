import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.utils import AverageMeter, mean_average_precision


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): Training data loader.
        device (torch.device): Device to move data to.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: returns logits with margin applied for ArcFace loss
        outputs = model(images, labels)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Train Loss: {loss_meter.avg}")
    return loss_meter.avg


def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Computes CrossEntropy loss and MAP@5.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): Validation data loader.
        device (torch.device): Device to move data to.

    Returns:
        tuple: (average_loss, map_at_5_score)
    """
    model.eval()
    loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Compute Loss (requires logits with margin)
            # We pass labels to the model to get the ArcFace logits
            logits_margin = model(images, labels)
            loss = F.cross_entropy(logits_margin, labels)
            loss_meter.update(loss.item(), images.size(0))

            # 2. Compute MAP@5 (requires raw similarity scores for ranking)
            # Get normalized embeddings (labels=None triggers inference mode in model)
            embeddings = model(images, labels=None)
            features = F.normalize(embeddings)

            # Get normalized head weights (class prototypes)
            head = model.head
            weights = F.normalize(head.weight)

            # Compute Cosine Similarity: (B, Out*K)
            cosine = F.linear(features, weights)

            # Handle Sub-centers: Reshape to (B, Out, K) and take max similarity across sub-centers
            cosine = cosine.view(-1, head.out_features, head.k)
            cosine, _ = torch.max(cosine, dim=2)

            # Get top 5 predictions based on similarity
            _, top_indices = torch.topk(cosine, k=5, dim=1)

            all_preds.append(top_indices.cpu())
            all_targets.append(labels.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MAP@5
    map_score = mean_average_precision(all_preds, all_targets, k=5)

    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation MAP@5: {map_score}")

    return loss_meter.avg, map_score


def inference(model, data_loader, device, idx_to_class, submission_path, use_tta=False):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (torch.utils.data.DataLoader): Test data loader.
        device (torch.device): Device to move data to.
        idx_to_class (np.ndarray or list): Mapping from class index to hotel_id.
        submission_path (str): Path to save the submission CSV.
        use_tta (bool): Whether to use Test-Time Augmentation (Horizontal Flip).
    """
    model.eval()
    results = []

    print("Starting inference...")
    with torch.no_grad():
        for images, image_ids in data_loader:
            images = images.to(device)

            # Get embeddings
            embeddings = model(images, labels=None)

            if use_tta:
                # Horizontal Flip TTA
                images_flip = torch.flip(images, dims=[3])
                embeddings_flip = model(images_flip, labels=None)
                # Average embeddings
                embeddings = (embeddings + embeddings_flip) / 2.0

            # Normalize features
            features = F.normalize(embeddings)

            # Normalize weights
            head = model.head
            weights = F.normalize(head.weight)

            # Compute Cosine Similarity
            cosine = F.linear(features, weights)
            cosine = cosine.view(-1, head.out_features, head.k)
            cosine, _ = torch.max(cosine, dim=2)

            # Get Top 5
            _, top_indices = torch.topk(cosine, k=5, dim=1)
            top_indices = top_indices.cpu().numpy()

            # Map indices to hotel IDs and format for submission
            for i, img_id in enumerate(image_ids):
                pred_indices = top_indices[i]
                # Map integer index to original hotel_id
                pred_labels = [str(idx_to_class[idx]) for idx in pred_indices]
                pred_string = " ".join(pred_labels)
                results.append({"image": img_id, "hotel_id": pred_string})

    # Save submission
    df = pd.DataFrame(results)
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
