import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import compute_metrics


def train_fn(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Executes one training epoch using Mixed Precision (AMP) and Gradient Accumulation.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Torch device (cuda/cpu).
        epoch: Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_fp16)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = len(dataloader)

    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        # Move inputs to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        response_mask_a = batch["response_mask_a"].to(device)

        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        response_mask_b = batch["response_mask_b"].to(device)

        scalars = batch["scalars"].to(device)
        targets = batch["target"].to(device)

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=Config.use_fp16):
            outputs = model(
                input_ids_a,
                attention_mask_a,
                response_mask_a,
                input_ids_b,
                attention_mask_b,
                response_mask_b,
                scalars,
            )
            loss = criterion(outputs, targets)

        # Scale Loss for Gradient Accumulation
        loss = loss / Config.accumulation_steps
        scaler.scale(loss).backward()

        # Gradient Accumulation Step
        if (i + 1) % Config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Track Loss (scale back up for logging)
        current_loss = loss.item() * Config.accumulation_steps
        total_loss += current_loss

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} Training Loss: {avg_loss}")

    return avg_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        device: Torch device.

    Returns:
        dict: Dictionary containing 'loss', 'log_loss', and 'accuracy'.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            response_mask_a = batch["response_mask_a"].to(device)

            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            response_mask_b = batch["response_mask_b"].to(device)

            scalars = batch["scalars"].to(device)
            targets = batch["target"].to(device)

            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                outputs = model(
                    input_ids_a,
                    attention_mask_a,
                    response_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    response_mask_b,
                    scalars,
                )
                loss = criterion(outputs, targets)

            total_loss += loss.item()

            # Apply softmax to get probabilities
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    metrics = compute_metrics(predictions, targets)
    metrics["loss"] = avg_loss

    print(f"Validation Loss: {metrics['loss']}")
    print(f"Validation Log Loss: {metrics['log_loss']}")
    print(f"Validation Accuracy: {metrics['accuracy']}")

    return metrics


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    Handles Test Time Augmentation (TTA) if enabled in Config.

    Args:
        model: The PyTorch model.
        dataloader: Test DataLoader.
        device: Torch device.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    model.eval()
    all_preds = []
    all_ids = []

    print("Starting Inference...")

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            response_mask_a = batch["response_mask_a"].to(device)

            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            response_mask_b = batch["response_mask_b"].to(device)

            scalars = batch["scalars"].to(device)
            ids = batch["id"]

            # 1. Forward Pass (Original: A vs B)
            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                outputs = model(
                    input_ids_a,
                    attention_mask_a,
                    response_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    response_mask_b,
                    scalars,
                )
            probs = torch.softmax(outputs, dim=1)

            # 2. Test Time Augmentation (Swap: B vs A)
            if Config.tta:
                # Swap scalars: [prompt, len_a, len_b] -> [prompt, len_b, len_a]
                scalars_swapped = scalars.clone()
                scalars_swapped[:, 1] = scalars[:, 2]
                scalars_swapped[:, 2] = scalars[:, 1]

                with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                    outputs_aug = model(
                        input_ids_b,
                        attention_mask_b,
                        response_mask_b,  # Swap inputs
                        input_ids_a,
                        attention_mask_a,
                        response_mask_a,
                        scalars_swapped,
                    )
                probs_aug = torch.softmax(outputs_aug, dim=1)

                # Swap probabilities back to original alignment:
                # Aug Output (B vs A): [Winner B, Winner A, Tie]
                # We want: [Winner A, Winner B, Tie]
                probs_aug_aligned = torch.zeros_like(probs_aug)
                probs_aug_aligned[:, 0] = probs_aug[
                    :, 1
                ]  # Winner A (was index 1 in aug)
                probs_aug_aligned[:, 1] = probs_aug[
                    :, 0
                ]  # Winner B (was index 0 in aug)
                probs_aug_aligned[:, 2] = probs_aug[:, 2]  # Tie

                # Average predictions
                probs = (probs + probs_aug_aligned) / 2.0

            all_preds.append(probs.cpu().numpy())
            all_ids.extend(ids)

    predictions = np.concatenate(all_preds)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "id": all_ids,
            "winner_model_a": predictions[:, 0],
            "winner_model_b": predictions[:, 1],
            "winner_tie": predictions[:, 2],
        }
    )

    # Save Submission
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")

    return submission


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience=2,
    save_path=None,
):
    """
    Orchestrates the training loop with Early Stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        epochs: Max epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.

    Returns:
        model: The model loaded with best weights.
    """
    if save_path is None:
        save_path = os.path.join(Config.working_dir, "best_model.pth")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, device, epoch + 1
        )

        # Validate
        val_metrics = eval_fn(model, val_loader, device)
        val_loss = val_metrics["loss"]

        # Early Stopping & Model Checkpointing
        if val_loss < best_loss:
            print(
                f"Validation Loss Improved ({best_loss} -> {val_loss}). Saving model..."
            )
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_loss}")
    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model
