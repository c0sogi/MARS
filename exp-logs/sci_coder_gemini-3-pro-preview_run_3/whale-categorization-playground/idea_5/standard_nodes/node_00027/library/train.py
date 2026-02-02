import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
from library.config import Config
from library.utils import AverageMeter, save_checkpoint, seed_everything
from library.dataset import get_loaders
from library.model import WhaleModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # WhaleModel returns logits when label is provided (CurricularFace)
        outputs = model(images, labels)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def extract_features(model, loader, device):
    """
    Extracts embeddings and labels for the entire dataset in the loader.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # WhaleModel returns normalized embeddings when label is None
            embeddings = model(images, label=None)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_embeddings, all_labels


def calculate_map5(query_feats, query_labels, gallery_feats, gallery_labels, device):
    """
    Computes MAP@5 using the gallery (train) to retrieve for queries (val).
    """
    # Move to GPU for fast matrix multiplication
    query_feats = query_feats.to(device)
    gallery_feats = gallery_feats.to(device)

    # Compute Cosine Similarity Matrix (Q x G)
    # Features are already normalized by the model
    sim_matrix = torch.mm(query_feats, gallery_feats.t())

    # Get Top 5 indices
    # We want the indices in the gallery that are closest to the query
    _, topk_indices = torch.topk(sim_matrix, k=5, dim=1)

    topk_indices = topk_indices.cpu().numpy()
    query_labels = query_labels.numpy()
    gallery_labels = gallery_labels.numpy()

    score_sum = 0.0
    n_queries = len(query_labels)

    for i in range(n_queries):
        true_label = query_labels[i]
        pred_indices = topk_indices[i]
        pred_labels = gallery_labels[pred_indices]

        # Calculate AP@5 for this query
        # We look for the first occurrence of the true label
        if true_label in pred_labels:
            # np.where returns a tuple, take first element array, then first index
            rank = np.where(pred_labels == true_label)[0][0]
            score_sum += 1.0 / (rank + 1)

    return score_sum / n_queries


def run_training():
    """
    Main training execution function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, _, num_classes = get_loaders(load_cached_data=True)

    # 3. Model
    print(f"Initializing model with backbone {Config.BACKBONE}...")
    model = WhaleModel(num_classes=num_classes)
    model = model.to(device)

    # 4. Optimization
    # CurricularFace requires a good optimizer setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_map5 = -1.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation (Retrieval-based)
        # Extract Gallery (Train) and Query (Val) features
        # Note: In a real large-scale scenario, we might use a fixed gallery or a subset.
        # Here, dataset is small enough to use full train set as gallery.
        gallery_feats, gallery_labels = extract_features(model, train_loader, device)
        query_feats, query_labels = extract_features(model, val_loader, device)

        val_map5 = calculate_map5(
            query_feats, query_labels, gallery_feats, gallery_labels, device
        )

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAP@5: {val_map5:.10f}"
        )

        # Checkpointing & Early Stopping
        is_best = val_map5 > best_map5
        if is_best:
            best_map5 = val_map5
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_map5": best_map5,
                },
                is_best=True,
                filepath=Config.MODEL_PATH,
            )
            print(f"New best model saved with MAP@5: {best_map5:.10f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    print(f"Training complete. Best Val MAP@5: {best_map5:.10f}")
