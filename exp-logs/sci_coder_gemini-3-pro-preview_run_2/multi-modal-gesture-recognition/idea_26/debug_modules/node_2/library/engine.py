import os
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm
from library import config, utils, loss, model

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    metrics_sum = {}
    num_batches = 0

    for batch in dataloader:
        # Move data to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}

        # Forward pass
        # Model returns list of outputs for each stage
        stage_outputs = model(skeleton, audio, targets["mask"])

        # Compute Loss
        loss_val, batch_metrics = criterion(stage_outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss_val.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.HYPERPARAMS["gradient_clip_val"]
        )

        optimizer.step()

        # Track metrics
        total_loss += loss_val.item()
        for k, v in batch_metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def validate(model, dataloader, criterion):
    """
    Runs validation loop and computes Levenshtein score.
    """
    model.eval()
    total_loss = 0.0
    metrics_sum = {}
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            lengths = batch["lengths"]

            # Forward pass
            stage_outputs = model(skeleton, audio, targets["mask"])

            # Compute Loss
            loss_val, batch_metrics = criterion(stage_outputs, targets)

            total_loss += loss_val.item()
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            num_batches += 1

            # --- Metric Calculation ---
            # Use Stage 3 output for predictions
            s3_out = stage_outputs[-1]
            logits = s3_out["cls"]  # (B, T, C)
            probs = torch.softmax(logits, dim=2)
            preds_indices = torch.argmax(probs, dim=2).cpu().numpy()  # (B, T)

            target_indices = targets["cls"].cpu().numpy()  # (B, T)

            # Process each sequence in batch
            for i in range(len(lengths)):
                length = lengths[i]

                # Get valid sequence (remove padding)
                pred_seq = preds_indices[i, :length]
                target_seq = target_indices[i, :length]

                # Collapse to gesture list
                # Note: We don't apply median filter in validation loop usually to save time,
                # or we can if we want metric to match inference exactly.
                # Let's apply simple collapse here.
                pred_gestures = utils.collapse_predictions(pred_seq)
                target_gestures = utils.collapse_predictions(target_seq)

                all_preds.append(pred_gestures)
                all_targets.append(target_gestures)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_metrics = (
        {k: v / num_batches for k, v in metrics_sum.items()} if num_batches > 0 else {}
    )

    # Compute Levenshtein Score
    lev_score = utils.compute_levenshtein_score(all_preds, all_targets)

    return avg_loss, lev_score, avg_metrics


def inference(model_path, dataloader, output_file):
    """
    Generates predictions for test set and saves to CSV.
    """
    print(f"Loading model from {model_path}...")
    # Initialize model structure
    net = model.HGGCRCN().to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()

    results = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in dataloader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            # Test batch has mask but no targets usually, but collate creates mask
            mask = batch["targets"]["mask"].to(device)
            sample_ids = batch["sample_ids"]
            lengths = batch["lengths"]

            # Forward pass
            stage_outputs = net(skeleton, audio, mask)

            # Use Stage 3
            s3_out = stage_outputs[-1]
            logits = s3_out["cls"]
            probs = torch.softmax(logits, dim=2)
            preds_indices = torch.argmax(probs, dim=2).cpu().numpy()  # (B, T)

            # Process batch
            for i in range(len(sample_ids)):
                sample_id = sample_ids[i]
                length = lengths[i]

                # Get raw prediction sequence
                raw_seq = preds_indices[i, :length]

                # --- Post-Processing ---
                # 1. Median Filter (Label-Space Smoothing)
                # Mode 'nearest' protects boundaries as requested
                kernel_size = config.HYPERPARAMS["median_filter_kernel"]
                if len(raw_seq) > 0:
                    smooth_seq = median_filter(
                        raw_seq, size=kernel_size, mode="nearest"
                    )
                else:
                    smooth_seq = raw_seq

                # 2. Collapse
                gesture_list = utils.collapse_predictions(smooth_seq)

                # Format: SessionID,Label1,Label2,...
                # Join labels with commas
                label_str = ",".join(map(str, gesture_list))
                results.append((sample_id, label_str))

    # Write to CSV
    # Format per prompt: Session00001,2,12,3
    print(f"Saving submission to {output_file}...")
    with open(output_file, "w") as f:
        for sample_id, label_str in results:
            if label_str:
                f.write(f"{sample_id},{label_str}\n")
            else:
                # Handle empty prediction case (no gestures found)
                f.write(f"{sample_id},\n")

    print("Inference complete.")


def run(
    train_loader,
    val_loader,
    test_loader,
    epochs=config.HYPERPARAMS["num_epochs"],
    patience=10,
):
    """
    Main training and evaluation routine.
    """
    # Initialize Model
    net = model.HGGCRCN().to(device)

    # Initialize Loss
    criterion = loss.HierarchicalLoss().to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.HYPERPARAMS["learning_rate"],
        weight_decay=config.HYPERPARAMS["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Setup Checkpointing
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    best_lev_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_metrics = train_one_epoch(
            net, train_loader, criterion, optimizer, epoch
        )

        # Validate
        val_loss, val_lev_score, val_metrics = validate(net, val_loader, criterion)

        # Step Scheduler
        scheduler.step(val_loss)

        # Print Metrics
        print(f"Epoch {epoch}/{epochs}")
        print(
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Levenshtein: {val_lev_score:.6f}"
        )
        # print(f"Train Metrics: {train_metrics}") # Optional detail

        # Early Stopping & Checkpointing
        if val_lev_score < best_lev_score:
            best_lev_score = val_lev_score
            patience_counter = 0
            torch.save(net.state_dict(), best_model_path)
            print(f"New best model saved! Score: {best_lev_score:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Run Inference with Best Model
    submission_file = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    inference(best_model_path, test_loader, submission_file)
