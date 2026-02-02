import time
import torch
import numpy as np
from library.config import CFG
from library.utils import AverageMeter, map_at_5


def train_fn(
    train_loader, model, criterion, optimizer, device, scheduler=None, epoch=0
):
    """
    Performs one epoch of training.

    Args:
        train_loader (DataLoader): Loader for training data.
        model (nn.Module): The model to train.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on.
        scheduler (object, optional): Learning rate scheduler.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    start = time.time()

    for step, batch in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - start)

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        batch_size = images.size(0)

        # Forward pass
        # When labels are provided, the model returns ArcFace logits
        logits = model(images, labels)

        loss = criterion(logits, labels)

        # Record loss
        losses.update(loss.item(), batch_size)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Note: Scheduler step is handled by the main loop (epoch-based)
        # or externally if it's a batch-based scheduler.

        # Measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Loss {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def extract_embeddings(loader, model, device, tta=False):
    """
    Extracts embeddings for a given dataset loader.
    Supports Test-Time Augmentation (Horizontal Flip).

    Args:
        loader (DataLoader): Data loader.
        model (nn.Module): Model to use for extraction.
        device (torch.device): Device.
        tta (bool): If True, applies horizontal flip TTA.

    Returns:
        tuple: (embeddings (np.array), ids (list))
    """
    model.eval()
    embeddings = []
    ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            # Get IDs if available (for gallery/query matching)
            if "id" in batch:
                ids.extend(batch["id"])

            # Forward pass (Original)
            # When labels are None, model returns normalized embeddings
            emb_org = model(images, labels=None)

            if tta:
                # Horizontal Flip TTA
                images_flipped = torch.flip(images, dims=[3])
                emb_flip = model(images_flipped, labels=None)

                # Average and re-normalize
                emb = (emb_org + emb_flip) / 2.0
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            else:
                emb = emb_org

            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    return embeddings, ids


def valid_fn(gallery_loader, val_loader, model, device):
    """
    Validation function using the Training set (or provided gallery) as the Reference.
    Computes MAP@5 for the validation set (Query) against the gallery.

    Args:
        gallery_loader (DataLoader): Loader for the gallery (usually training set).
        val_loader (DataLoader): Loader for the validation set.
        model (nn.Module): The model.
        device (torch.device): Device.

    Returns:
        float: The MAP@5 score.
    """
    print("Extracting embeddings for Validation (Query)...")
    val_embeddings, val_ids = extract_embeddings(
        val_loader, model, device, tta=CFG.tta_flips
    )

    print("Extracting embeddings for Gallery (Train)...")
    # We use the gallery_loader to get the reference embeddings.
    train_embeddings, train_ids = extract_embeddings(
        gallery_loader, model, device, tta=CFG.tta_flips
    )

    print("Computing Similarity Matrix...")
    # Compute Cosine Similarity: Query (Val) x Gallery (Train)^T
    # Embeddings are already normalized, so dot product is cosine similarity.
    similarity_matrix = np.dot(val_embeddings, train_embeddings.T)

    print("Calculating MAP@5...")
    predictions = []

    # For each validation query
    for i in range(len(val_ids)):
        # Get scores for this query
        scores = similarity_matrix[i]

        # Sort indices descending
        sorted_indices = np.argsort(scores)[::-1]

        pred_ids = []
        seen = set()

        # Iterate through sorted indices to find top 5 unique IDs
        for idx in sorted_indices:
            pred_id = train_ids[idx]

            if pred_id not in seen:
                pred_ids.append(pred_id)
                seen.add(pred_id)

            if len(pred_ids) == 5:
                break

        predictions.append(pred_ids)

    # Compute Metric
    score = map_at_5(val_ids, predictions)

    print(f"Validation MAP@5: {score}")

    return score
