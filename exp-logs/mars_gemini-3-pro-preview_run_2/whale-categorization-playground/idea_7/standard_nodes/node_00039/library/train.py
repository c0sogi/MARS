import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config, seed_everything
from library.utils import AverageMeter, mapk
from library.loss import ArcFaceLoss
from library.dataset import get_train_val_loaders, get_label_mapping
from library.model import WhaleModel


def train_one_epoch(loader, model, loss_fn, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_fn.train()

    losses = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (Backbone)
        embeddings = model(images)

        # Forward pass (Head + Loss)
        loss = loss_fn(embeddings, labels)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(loader, model, loss_fn, device, threshold=0.5):
    """
    Validates the model using MAP@5.
    Uses ArcFace class centers as the gallery for known whales.
    """
    model.eval()
    loss_fn.eval()

    # 1. Get Mapping
    # id2idx maps Label -> Int. We need Int -> Label for predictions.
    _, idx2id = get_label_mapping()

    # 2. Extract Validation Embeddings
    val_embeddings = []
    val_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            emb = model(images)
            val_embeddings.append(emb.cpu())
            val_targets.extend(labels.numpy())

    val_embeddings = torch.cat(val_embeddings, dim=0)
    # Normalize validation embeddings
    val_embeddings = F.normalize(val_embeddings, p=2, dim=1)

    # 3. Get Class Centers (Gallery)
    # loss_fn.weight is (Num_Classes, Emb_Size)
    with torch.no_grad():
        centers = loss_fn.weight.clone().cpu()
        centers = F.normalize(centers, p=2, dim=1)

    # 4. Compute Similarity Matrix
    # Shape: (N_val, N_classes)
    sim_matrix = torch.matmul(val_embeddings, centers.T)

    # 5. Generate Predictions with Thresholding
    predicted_labels = []
    actual_labels = []

    for i in range(len(val_targets)):
        # Ground Truth
        target_idx = val_targets[i]
        if target_idx == -1:
            actual = "new_whale"
        else:
            actual = idx2id[target_idx]
        actual_labels.append([actual])

        # Prediction
        # Get top 5 candidates from known classes
        scores, indices = torch.topk(sim_matrix[i], k=5)
        scores = scores.numpy()
        indices = indices.numpy()

        preds = []
        new_whale_added = False

        for score, idx in zip(scores, indices):
            # If we haven't added new_whale yet and confidence is low, add it
            if not new_whale_added and score < threshold:
                preds.append("new_whale")
                new_whale_added = True

            # Stop if we have 5 predictions
            if len(preds) >= 5:
                break

            # Add the known whale
            preds.append(idx2id[idx])

        # Fill remaining slots if needed
        if len(preds) < 5 and not new_whale_added:
            preds.append("new_whale")

        # Ensure exactly 5 predictions (truncate if logic above overfilled)
        predicted_labels.append(preds[:5])

    # 6. Compute MAP@5
    score = mapk(actual_labels, predicted_labels, k=5)

    return score


def run_training_phase():
    """
    Main orchestration function for the Dual-Backbone Ensemble training.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training on device: {device}")

    # Iterate over defined models (e.g., EfficientNet-B4, EfficientNet-V2-M)
    for model_cfg in Config.MODEL_CONFIGS:
        model_name = model_cfg["name"]
        backbone_name = model_cfg["backbone"]
        emb_size = model_cfg["embedding_size"]

        print(f"\n{'='*40}")
        print(f"Training Model: {model_name} ({backbone_name})")
        print(f"{'='*40}")

        # Initialize Model and Loss
        # We re-initialize for the first stage.
        model = WhaleModel(backbone_name, pretrained=True, embedding_size=emb_size)
        model = model.to(device)

        loss_fn = ArcFaceLoss(
            in_features=emb_size,
            out_features=Config.N_CLASSES,
            s=Config.ARCFACE_SCALE,
            m=Config.ARCFACE_MARGIN,
        )
        loss_fn = loss_fn.to(device)

        # Iterate over Progressive Resizing Stages
        # Stage 1: 256x256, Stage 2: 384x384
        for stage_idx, stage_cfg in enumerate(Config.STAGES):
            resolution = stage_cfg["resolution"]
            epochs = stage_cfg["epochs"]

            print(
                f"\n--- Stage {stage_idx+1}: Resolution {resolution}x{resolution} ---"
            )

            # If this is not the first stage, we must ensure we are continuing
            # from the best weights of the previous stage.
            if stage_idx > 0:
                prev_resolution = Config.STAGES[stage_idx - 1]["resolution"]
                checkpoint_path = os.path.join(
                    Config.WORKING_DIR, f"{model_name}_{prev_resolution}_best.pth"
                )
                if os.path.exists(checkpoint_path):
                    print(
                        f"Loading checkpoint from {checkpoint_path} for fine-tuning..."
                    )
                    checkpoint = torch.load(
                        checkpoint_path, map_location=device, weights_only=False
                    )
                    model.load_state_dict(checkpoint["model_state_dict"])
                    loss_fn.load_state_dict(checkpoint["loss_state_dict"])
                else:
                    print(
                        f"Warning: Previous checkpoint not found. Starting from scratch."
                    )

            # Prepare DataLoaders for current resolution
            train_loader, val_loader = get_train_val_loaders(
                resolution=resolution,
                batch_size=Config.BATCH_SIZE,
                load_cached_data=True,
            )

            # Optimizer & Scheduler
            # Re-initialize optimizer for each stage to reset momentum
            optimizer = AdamW(
                list(model.parameters()) + list(loss_fn.parameters()),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            # Training Loop
            best_map = 0.0
            patience = 5
            patience_counter = 0

            for epoch in range(epochs):
                start_time = time.time()

                # Train
                train_loss = train_one_epoch(
                    train_loader, model, loss_fn, optimizer, device, epoch
                )

                # Validate
                val_map = validate(
                    val_loader,
                    model,
                    loss_fn,
                    device,
                    threshold=Config.CONFIDENCE_THRESHOLD,
                )

                # Step Scheduler
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

                elapsed = time.time() - start_time

                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"LR: {current_lr:.2e} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val MAP@5: {val_map:.6f} | "
                    f"Time: {elapsed:.0f}s"
                )

                # Checkpointing
                if val_map > best_map:
                    best_map = val_map
                    patience_counter = 0

                    save_path = os.path.join(
                        Config.WORKING_DIR, f"{model_name}_{resolution}_best.pth"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "loss_state_dict": loss_fn.state_dict(),
                            "best_map": best_map,
                        },
                        save_path,
                    )
                    print(
                        f"  >>> New Best Score! Model saved to {os.path.basename(save_path)}"
                    )
                else:
                    patience_counter += 1

                # Early Stopping
                if patience_counter >= patience:
                    print(
                        f"  >>> Early stopping triggered after {patience} epochs of no improvement."
                    )
                    break

            print(f"Stage {stage_idx+1} completed. Best MAP@5: {best_map:.6f}")

            # Clean up memory
            del train_loader, val_loader, optimizer, scheduler
            torch.cuda.empty_cache()

        # Clean up model
        del model, loss_fn
        torch.cuda.empty_cache()

    print("\nAll training phases completed.")
