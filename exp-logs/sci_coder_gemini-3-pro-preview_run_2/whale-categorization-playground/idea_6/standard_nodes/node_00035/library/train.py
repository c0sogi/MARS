import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, AverageMeter, map_5, save_checkpoint
from library.dataset import WhaleDataset, create_id_map
from library.loss import ArcFaceLoss
from library.model import WhaleModel


def train_fn(dataloader, model, criterion, optimizer, device, epoch):
    """
    Training function for one epoch.
    """
    model.train()
    # The ArcFace loss layer contains trainable weights (class centers), so it must be in train mode too
    criterion.train()

    loss_meter = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: Get embeddings
        embeddings = model(images)

        # Calculate loss: ArcFace takes embeddings and ground truth labels
        loss = criterion(embeddings, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def eval_fn(dataloader, model, criterion, device):
    """
    Validation function.
    Computes MAP@5 by comparing image embeddings to ArcFace class centers.
    """
    model.eval()
    criterion.eval()

    all_preds = []
    all_targets = []

    # Pre-compute normalized class centers (weights from ArcFaceLoss)
    # Shape: (Num_Classes, Embedding_Dim)
    with torch.no_grad():
        class_centers = F.normalize(criterion.weight, p=2, dim=1)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Get image embeddings
            embeddings = model(images)
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Compute Cosine Similarity: (Batch, Emb) @ (Classes, Emb).T -> (Batch, Classes)
            logits = torch.matmul(embeddings, class_centers.T)

            # Get Top 5 predictions
            _, top_indices = torch.topk(logits, k=5, dim=1)

            all_preds.extend(top_indices.cpu().numpy())
            all_targets.extend(labels.numpy())

    # Calculate MAP@5
    score = map_5(all_preds, all_targets)
    return score


def run_training():
    """
    Main training loop implementing Progressive Resolution.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 1. Create ID Map (Excluding new_whale)
    id_map = create_id_map(Config.TRAIN_CSV)
    Config.NUM_CLASSES = len(id_map)
    print(f"Number of classes (excluding new_whale): {Config.NUM_CLASSES}")

    # 2. Initialize Model
    model = WhaleModel(embedding_size=Config.EMBEDDING_SIZE, pretrained=True)
    if Config.USE_GRADIENT_CHECKPOINTING:
        model.enable_gradient_checkpointing()
    model.to(device)

    # 3. Initialize Loss (ArcFace)
    criterion = ArcFaceLoss(
        in_features=Config.EMBEDDING_SIZE,
        out_features=Config.NUM_CLASSES,
        s=Config.ARC_S,
        m=Config.ARC_M,
    ).to(device)

    # 4. Optimizer & Scheduler
    # Optimize both model parameters and ArcFace loss parameters (class weights)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    # State tracking
    best_score = 0.0
    patience_counter = 0
    current_image_size = 0
    train_loader = None
    val_loader = None

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # -----------------------------------------------------------------
        # Progressive Resolution Logic
        # -----------------------------------------------------------------
        # Determine target resolution for this epoch
        if epoch <= Config.PHASE_1_EPOCHS:
            target_size = Config.IMG_SIZE_START
            phase_name = "Phase 1 (Warm-up)"
        else:
            target_size = Config.IMG_SIZE_FINAL
            phase_name = "Phase 2 (Fine-tuning)"

        # Re-initialize DataLoaders if resolution changes
        if target_size != current_image_size:
            print(
                f"\n[Resolution Switch] Changing resolution to {target_size}x{target_size} ({phase_name})"
            )
            current_image_size = target_size

            # Train Dataset
            train_dataset = WhaleDataset(
                csv_path=Config.TRAIN_CSV,
                subset_name="train",
                image_size=current_image_size,
                id_map=id_map,
                mode="train",
                filter_new_whale=True,  # Strictly train on known whales
                load_cached_data=True,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

            # Validation Dataset
            val_dataset = WhaleDataset(
                csv_path=Config.VAL_CSV,
                subset_name="val",
                image_size=current_image_size,
                id_map=id_map,
                mode="val",
                filter_new_whale=True,  # Validate on known whales only
                load_cached_data=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

        # -----------------------------------------------------------------
        # Training & Validation
        # -----------------------------------------------------------------
        avg_loss = train_fn(train_loader, model, criterion, optimizer, device, epoch)
        val_score = eval_fn(val_loader, model, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} [{phase_name}] - "
            f"Loss: {avg_loss:.6f} - MAP@5: {val_score:.6f} - "
            f"LR: {current_lr:.2e} - Time: {elapsed:.0f}s"
        )

        # -----------------------------------------------------------------
        # Checkpointing & Early Stopping
        # -----------------------------------------------------------------
        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            patience_counter = 0
            print(f"  >>> New Best Score! Saving model...")
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                # We also save the ArcFace weights as they represent the learned class centers
                "arcface_dict": criterion.state_dict(),
            },
            is_best,
            filename=f"{Config.WORKING_DIR}/checkpoint_last.pth",
        )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best MAP@5: {best_score:.6f}")


if __name__ == "__main__":
    # This block is for local testing only, will not be executed by the pipeline import
    run_training()
