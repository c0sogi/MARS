import torch
import numpy as np
import sys
from library.utils import calculate_metric, save_checkpoint, ModelEma


def train_one_epoch(model, loader, optimizer, device, loss_fn, scaler, ema_model=None):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP).

    Args:
        model (torch.nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to compute on.
        loss_fn (nn.Module): Loss function (MultiTaskLoss).
        scaler (GradScaler): AMP GradScaler.
        ema_model (ModelEma, optional): EMA model wrapper for weight averaging.

    Returns:
        dict: Average losses for the epoch.
    """
    model.train()
    running_losses = {}
    count = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        # targets is a dictionary of tensors, move all to device
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss, loss_dict = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema_model:
            ema_model.update(model)

        # Accumulate losses for logging
        count += 1
        for k, v in loss_dict.items():
            val = v
            if isinstance(v, torch.Tensor):
                val = v.item()
            running_losses[k] = running_losses.get(k, 0.0) + val

    # Compute average losses
    avg_losses = {k: v / count for k, v in running_losses.items()}
    return avg_losses


def validate(model, loader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate (usually EMA model).
        loader (DataLoader): Validation data loader.
        device (torch.device): Device to compute on.
        loss_fn (nn.Module): Loss function.

    Returns:
        tuple: (avg_losses dict, auc_score float)
    """
    model.eval()
    running_losses = {}
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)
            loss, loss_dict = loss_fn(outputs, targets)

            # Accumulate losses
            count += 1
            for k, v in loss_dict.items():
                val = v
                if isinstance(v, torch.Tensor):
                    val = v.item()
                running_losses[k] = running_losses.get(k, 0.0) + val

            # Store predictions for AUC calculation
            # We use the 'main' head probabilities
            main_logits = outputs["main"]
            main_probs = torch.softmax(main_logits, dim=1)

            all_preds.append(main_probs.cpu().numpy())
            all_targets.append(targets["main"].cpu().numpy())

    # Compute average losses
    avg_losses = {k: v / count for k, v in running_losses.items()}

    # Calculate Metric (Mean Column-wise ROC AUC)
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        auc_score = calculate_metric(all_targets, all_preds)
    else:
        auc_score = 0.5

    return avg_losses, auc_score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    config,
    save_path,
    scheduler=None,
):
    """
    Executes the full training loop with Early Stopping, EMA, and Logging.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device.
        config (Config): Configuration object.
        save_path (str): Path to save the best model checkpoint.
        scheduler (optional): Learning rate scheduler.

    Returns:
        float: Best validation AUC score achieved.
    """
    # Initialize Scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Initialize EMA if configured
    ema_model = None
    if config.USE_EMA:
        print(f"Initializing Model EMA with decay {config.EMA_DECAY}")
        ema_model = ModelEma(model, decay=config.EMA_DECAY, device=device)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(1, config.EPOCHS + 1):
        # 1. Train Step
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, scaler, ema_model
        )

        # 2. Validation Step
        # Use EMA model for validation if available (provides more stable predictions)
        eval_model = ema_model.apply_shadow() if ema_model else model
        val_losses, val_auc = validate(eval_model, val_loader, device, loss_fn)

        # 3. Scheduler Step
        if scheduler:
            scheduler.step()

        # 4. Logging
        # Printing full precision as requested
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print(
            f"Train Loss: {train_losses['loss_total']:.6f} [Main: {train_losses['loss_main']:.6f}]"
        )
        print(
            f"Val Loss:   {val_losses['loss_total']:.6f} [Main: {val_losses['loss_main']:.6f}]"
        )
        print(f"Val AUC:    {val_auc}")

        # 5. Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            # Save the best model
            # We save the state_dict of the evaluation model (EMA if active)
            state_dict = eval_model.state_dict()

            checkpoint = {
                "epoch": epoch,
                "state_dict": state_dict,
                "auc": best_auc,
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, save_path)
            print(f"Saved new best model to {save_path}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered. Training finished.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc
