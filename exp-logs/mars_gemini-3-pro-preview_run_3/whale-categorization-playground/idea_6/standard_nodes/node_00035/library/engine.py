import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from library.config import Config
from library.utils import AverageMeter
from library.inference_utils import l2_normalize, run_inference_pipeline
from library.dataset import get_inference_gallery_loader, get_test_loader


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Executes one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    # Iterate over the data loader
    # Note: Progress bar suppressed as per instructions for cleaner logs
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        # When targets are provided, the model returns ArcFace logits
        logits = model(images, targets)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def extract_features(model, loader, device):
    """
    Extracts embeddings for validation (in-memory, no caching).
    Used during the training loop where model weights change frequently.
    """
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            # Forward pass without targets returns embeddings
            emb = model(images)

            embeddings.append(emb.cpu())
            labels.append(targets)

    # Concatenate and Normalize
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = l2_normalize(embeddings)

    # Handle labels (could be tensor or list)
    if isinstance(labels[0], torch.Tensor):
        labels = torch.cat(labels, dim=0).numpy()
    else:
        labels = np.concatenate(labels, axis=0)

    return embeddings.numpy(), labels


def calculate_map5(query_feats, query_labels, gallery_feats, gallery_labels):
    """
    Computes MAP@5 for the validation set against the training gallery.
    """
    # Compute Cosine Similarity Matrix: (N_query, N_gallery)
    # feats are already L2 normalized, so dot product is cosine similarity
    sims = np.dot(query_feats, gallery_feats.T)

    scores = []

    # Iterate through each query
    for i in range(len(query_labels)):
        q_label = query_labels[i]

        # Get indices of gallery items sorted by similarity (descending)
        # We only need the top candidates, but full sort is fast enough for N=6k
        sorted_indices = np.argsort(sims[i])[::-1]

        # Retrieve top 5 unique IDs
        top_ids = []
        seen_ids = set()

        for idx in sorted_indices:
            g_label = gallery_labels[idx]

            if g_label not in seen_ids:
                top_ids.append(g_label)
                seen_ids.add(g_label)

            if len(top_ids) >= 5:
                break

        # Calculate Score
        if q_label in top_ids:
            rank = top_ids.index(q_label) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)

    return np.mean(scores)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    label_encoder,
    epochs=Config.NUM_EPOCHS,
    patience=5,
):
    """
    Main training loop with validation and early stopping.
    """
    best_map = 0.0
    patience_counter = 0

    # Load a clean gallery loader (Train set without augmentation) for validation
    gallery_loader = get_inference_gallery_loader(
        load_cached_data=True, debug=Config.DEBUG
    )

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # 1. Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # 2. Step Scheduler
        if scheduler:
            scheduler.step()

        # 3. Validate (Retrieval Task)
        # Extract features for Query (Val) and Gallery (Train)
        # We do this every epoch to track MAP@5
        val_feats, val_labels = extract_features(model, val_loader, device)
        gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

        val_map = calculate_map5(val_feats, val_labels, gallery_feats, gallery_labels)

        # 4. Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{epochs}] LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val MAP@5: {val_map:.6f}"
        )

        # 5. Early Stopping & Checkpointing
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> New Best Model Saved! (MAP@5: {best_map:.6f})")
        else:
            patience_counter += 1
            print(f"  >>> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MAP@5: {best_map:.6f}")

    # Load best weights for inference
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    return model


def predict(model, train_loader, label_encoder, device):
    """
    Generates predictions for the test set using the full inference pipeline
    (Query Expansion + Re-ranking).
    """
    print("Generating submission...")
    test_loader = get_test_loader(debug=Config.DEBUG)

    # Use the sophisticated pipeline from inference_utils
    # We force load_cached=False for the query/gallery embeddings here to ensure
    # we use the features from the *final best model*, not stale cache from previous runs.
    run_inference_pipeline(
        model=model,
        train_loader=train_loader,  # This acts as the gallery
        test_loader=test_loader,
        label_encoder=label_encoder,
        device=device,
        load_cached=False,
    )
