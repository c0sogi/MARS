import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from library.config import CFG


def masked_mae_loss(pred, target, mask):
    """
    Calculates Mean Absolute Error loss, ignoring padded time steps.

    Args:
        pred: (B, C, L)
        target: (B, C, L)
        mask: (B, L) -> will be unsqueezed to (B, 1, L)
    """
    mask = mask.unsqueeze(1)  # (B, 1, L)
    # Expand mask to cover channels if necessary, though broadcasting handles it
    loss = torch.abs(pred - target) * mask
    # Sum over all dimensions and divide by sum of mask elements * channels
    # Add epsilon to avoid division by zero
    num_valid = mask.sum() * pred.shape[1]
    return loss.sum() / (num_valid + 1e-8)


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (features, targets, mask, _) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Forward pass: returns [main, aux1, aux2]
        outputs = model(features)

        # 1. Main Head Loss (Full Resolution)
        loss_main = masked_mae_loss(outputs[0], targets, mask)
        total_loss = loss_main * CFG.LOSS_WEIGHTS[0]

        # 2. Auxiliary Heads Loss (Scaled Deep Supervision)
        # outputs[1] is aux1 (1/2 res), outputs[2] is aux2 (1/4 res)

        # Aux 1
        if len(outputs) > 1 and CFG.LOSS_WEIGHTS[1] > 0:
            aux1_pred = outputs[1]
            target_aux1 = F.adaptive_avg_pool1d(targets, aux1_pred.shape[-1])
            # Nearest neighbor for mask to keep it binary-ish (0 or 1)
            mask_aux1 = F.interpolate(
                mask.unsqueeze(1), size=aux1_pred.shape[-1], mode="nearest"
            ).squeeze(1)
            loss_aux1 = masked_mae_loss(aux1_pred, target_aux1, mask_aux1)
            total_loss += loss_aux1 * CFG.LOSS_WEIGHTS[1]

        # Aux 2
        if len(outputs) > 2 and CFG.LOSS_WEIGHTS[2] > 0:
            aux2_pred = outputs[2]
            target_aux2 = F.adaptive_avg_pool1d(targets, aux2_pred.shape[-1])
            mask_aux2 = F.interpolate(
                mask.unsqueeze(1), size=aux2_pred.shape[-1], mode="nearest"
            ).squeeze(1)
            loss_aux2 = masked_mae_loss(aux2_pred, target_aux2, mask_aux2)
            total_loss += loss_aux2 * CFG.LOSS_WEIGHTS[2]

        # Backward
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRADIENT_CLIP)

        optimizer.step()

        running_loss += total_loss.item()

    # Step scheduler after epoch (CosineAnnealingLR usually steps per epoch)
    if scheduler is not None:
        scheduler.step()

    return running_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for features, targets, mask, _ in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(features)

            # For validation, we only care about the main output performance
            main_output = outputs[0]

            loss = masked_mae_loss(main_output, targets, mask)
            running_loss += loss.item()

    return running_loss / len(dataloader)


def train_model(model, train_loader, val_loader):
    device = CFG.DEVICE
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.T_MAX, eta_min=CFG.ETA_MIN
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")
    print(f"Epochs: {CFG.EPOCHS}, Batch Size: {CFG.BATCH_SIZE}")
    print(f"Loss Weights: {CFG.LOSS_WEIGHTS}")

    for epoch in range(CFG.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{CFG.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CFG.BEST_MODEL_PATH)
            print(f"  -> New best model saved! (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{CFG.PATIENCE}")

        if patience_counter >= CFG.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")

    # Load best model for return
    model.load_state_dict(torch.load(CFG.BEST_MODEL_PATH, map_location=device))
    return model
