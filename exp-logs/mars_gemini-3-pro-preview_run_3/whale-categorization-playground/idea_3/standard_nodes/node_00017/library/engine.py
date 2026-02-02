import os
import torch
import torch.nn.functional as F
import numpy as np
from library.utils import AverageMeter, calc_map5, save_checkpoint
from library.config import CFG
from library.rerank import re_ranking


def train_fn(dataloader, model, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # In train mode, the model returns cosine logits from the CurricularFaceHead
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def eval_fn(val_loader, gallery_loader, model, device, id_map):
    """
    Performs validation by computing MAP@5 using a retrieval approach (Query vs Gallery).
    Cite solution_lesson_node_00016: Validate using inference pipeline, not training objective.
    """
    model.eval()

    # Create inverse map: int -> str
    idx_to_id = {v: k for k, v in id_map.items()}

    # 1. Extract Gallery Features (Train Set)
    gallery_feats = []
    gallery_labels = []
    with torch.no_grad():
        for images, labels in gallery_loader:
            images = images.to(device)
            feats = model(images)
            gallery_feats.append(feats)
            gallery_labels.extend(labels.numpy())

    gallery_feats = torch.cat(gallery_feats, dim=0)
    # Normalize features (important for cosine distance / re-ranking)
    gallery_feats = F.normalize(gallery_feats, p=2, dim=1)

    # 2. Extract Query Features (Val Set)
    query_feats = []
    query_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            feats = model(images)
            query_feats.append(feats)
            query_labels.extend(labels.numpy())

    query_feats = torch.cat(query_feats, dim=0)
    query_feats = F.normalize(query_feats, p=2, dim=1)

    # 3. Compute Distance Matrix
    # Use re-ranking if enabled to match inference
    if CFG.use_reranking:
        dist_mat = re_ranking(
            query_feats.cpu().numpy(),
            gallery_feats.cpu().numpy(),
            k1=CFG.k1,
            k2=CFG.k2,
            lambda_value=CFG.lambda_value,
        )
    else:
        # Cosine distance = 1 - cosine_similarity
        sim_mat = torch.mm(query_feats, gallery_feats.t())
        dist_mat = 1.0 - sim_mat.cpu().numpy()

    # 4. Generate Predictions
    predictions = []
    ground_truths = []

    # Pre-sort distances for efficiency
    # argsort along axis 1 (gallery items)
    sorted_indices = np.argsort(dist_mat, axis=1)

    for i in range(len(query_labels)):
        # Ground Truth
        true_idx = query_labels[i]
        if true_idx in idx_to_id:
            true_label = idx_to_id[true_idx]
        else:
            true_label = "new_whale"
        ground_truths.append(true_label)

        # Predictions
        preds = []

        # Get distances for this query
        dists = dist_mat[i]
        indices = sorted_indices[i]

        # Check nearest neighbor distance for 'new_whale' threshold
        min_dist = dists[indices[0]]

        if min_dist > CFG.new_whale_threshold:
            preds.append("new_whale")

        for idx in indices:
            train_idx = gallery_labels[idx]
            label_str = idx_to_id.get(train_idx, "new_whale")

            if label_str not in preds:
                preds.append(label_str)

            if len(preds) >= 5:
                break

        # Fill with new_whale if needed
        while len(preds) < 5:
            preds.append("new_whale")

        predictions.append(preds[:5])

    # Calculate MAP@5
    score = calc_map5(ground_truths, predictions)
    return score


def fit(
    model,
    train_loader,
    val_loader,
    gallery_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience=5,
    save_path=None,
):
    """
    Main training loop with early stopping and checkpointing.
    """
    if save_path is None:
        save_path = CFG.model_path

    best_score = 0.0
    patience_counter = 0

    # Retrieve id_map from the validation dataset to decode predictions
    if hasattr(val_loader.dataset, "get_id_map"):
        id_map = val_loader.dataset.get_id_map()
    else:
        id_map = val_loader.dataset.id_map

    for epoch in range(epochs):
        # Train Step
        train_loss = train_fn(train_loader, model, criterion, optimizer, device)

        # Validation Step
        val_score = eval_fn(val_loader, gallery_loader, model, device, id_map)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Logging (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val MAP@5: {val_score}"
        )

        # Checkpointing
        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            patience_counter = 0
        else:
            patience_counter += 1

        # Save model state and id_map
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "id_map": id_map,
            },
            is_best,
            filepath=save_path,
        )

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return best_score


def predict_and_submit(model, train_loader, test_loader, device, id_map, save_path):
    """
    Generates predictions for the test set using k-Reciprocal Re-ranking and saves to CSV.
    """
    model.eval()

    # 1. Extract Gallery Features (Training Set)
    gallery_feats = []
    gallery_labels = []

    # We use the train_loader to build the gallery
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            feats = model(images)  # Returns embeddings
            gallery_feats.append(feats.cpu())
            gallery_labels.extend(labels.numpy())

    gallery_feats = torch.cat(gallery_feats, dim=0)

    # 2. Extract Probe Features (Test Set)
    probe_feats = []
    image_names = []

    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            feats = model(images)
            probe_feats.append(feats.cpu())
            image_names.extend(names)

    probe_feats = torch.cat(probe_feats, dim=0)

    # 3. Compute Re-ranked Distance Matrix
    # probFea: (M, D), galFea: (N, D) -> dist: (M, N)
    print("Computing re-ranked distance matrix...")
    dist_mat = re_ranking(
        probe_feats.numpy(),
        gallery_feats.numpy(),
        k1=CFG.k1,
        k2=CFG.k2,
        lambda_value=CFG.lambda_value,
    )

    # 4. Generate Predictions
    idx_to_id = {v: k for k, v in id_map.items()}

    print(f"Generating submission file at {save_path}...")
    with open(save_path, "w") as f:
        f.write("Image,Id\n")

        for i in range(len(image_names)):
            img_name = image_names[i]
            dists = dist_mat[i]

            # Get indices of sorted distances (ascending, smallest distance first)
            sorted_indices = np.argsort(dists)

            # Distance to the nearest neighbor
            min_dist = dists[sorted_indices[0]]

            preds = []

            # Strategy:
            # If the nearest neighbor is too far (dist > threshold), predict 'new_whale' first.
            if min_dist > CFG.new_whale_threshold:
                preds.append("new_whale")

            # Fill remaining slots with nearest neighbors
            for idx in sorted_indices:
                train_label_idx = gallery_labels[idx]

                # Handle case where training label might not be in map (unlikely)
                label_str = idx_to_id.get(train_label_idx, "new_whale")

                if label_str not in preds:
                    preds.append(label_str)

                if len(preds) >= 5:
                    break

            # If 'new_whale' wasn't added at start but we still have space, append it
            # This covers the case where the match is decent but not perfect,
            # or simply as a fallback option.
            if "new_whale" not in preds and len(preds) < 5:
                preds.append("new_whale")

            # Ensure we have exactly 5 predictions (though logic above guarantees <= 5)
            # If we somehow have < 5 (e.g. very small gallery), pad with new_whale
            while len(preds) < 5:
                preds.append("new_whale")

            # Truncate to 5 strictly
            preds = preds[:5]

            f.write(f"{img_name},{' '.join(preds)}\n")
