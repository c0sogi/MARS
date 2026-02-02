import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from library.config import Config
from library.utils import set_seed, compute_levenshtein, median_filter, rle_decode
from library.data_loader import get_dataloaders
from library.model import BAMPNet


def train_epoch(model, loader, optimizer, criterion_class, criterion_boundary, device):
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(skeleton, audio, lengths, mask)
        logits = outputs["logits"]  # (B, T, NumClasses+1)

        # Flatten: (B*T, C)
        logits_flat = logits.view(-1, logits.shape[-1])
        labels_flat = labels.view(-1)

        # Compute Loss
        loss = criterion_class(logits_flat, labels_flat)

        # Backward
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_VAL)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0,
    }


def validate(model, loader, gt_map, device):
    """
    Runs validation and computes Levenshtein Error Rate.
    """
    model.eval()
    total_dist = 0
    total_gestures = 0

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]
            ids = batch["ids"]

            outputs = model(skeleton, audio, lengths, mask)
            logits = outputs["logits"]  # (B, T, C)

            # Get predictions
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2).cpu().numpy()  # (B, T)

            for i, sample_id in enumerate(ids):
                # Get valid length
                valid_len = lengths[i]
                sample_pred = preds[i, :valid_len]

                # Post-processing
                # 1. Median Filter
                sample_pred = median_filter(sample_pred, window_size=5)

                # 2. RLE Decode
                pred_seq = rle_decode(
                    sample_pred,
                    background_class_id=Config.BACKGROUND_CLASS_ID,
                    min_duration=5,
                )

                # Get Ground Truth
                gt_seq = gt_map.get(sample_id, [])

                # Compute Distance
                dist = compute_levenshtein(pred_seq, gt_seq)

                total_dist += dist
                total_gestures += len(gt_seq)

    # Avoid division by zero
    if total_gestures == 0:
        return 0.0

    return total_dist / total_gestures


def generate_submission(model, loader, device, output_path):
    """
    Generates submission file for test set.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]
            ids = batch["ids"]

            outputs = model(skeleton, audio, lengths, mask)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=2).cpu().numpy()

            for i, sample_id in enumerate(ids):
                valid_len = lengths[i]
                sample_pred = preds[i, :valid_len]

                # Post-processing
                sample_pred = median_filter(sample_pred, window_size=5)
                pred_seq = rle_decode(
                    sample_pred,
                    background_class_id=Config.BACKGROUND_CLASS_ID,
                    min_duration=5,
                )

                # Format string: "Label1,Label2,..."
                pred_str = ",".join(map(str, pred_seq))
                results.append((sample_id, pred_str))

    # Write to CSV
    with open(output_path, "w", newline="") as f:
        # The competition format usually doesn't have a header,
        # but based on the prompt "Session00001,2,12,3", it's ID,Sequence.
        # The prompt submission format section shows: "Session00001,2,12,3"
        # It does NOT explicitly ask for a header.
        # However, standard CSV writers might be safer.
        # We will write line by line to match the example exactly.
        writer = csv.writer(f)
        for sample_id, pred_str in results:
            # We write as two columns: ID, Sequence(comma-separated)
            # But the example "Session00001,2,12,3" looks like all in one line separated by commas.
            # "Session00001" is the first element, then the labels.
            row = [sample_id] + pred_str.split(",") if pred_str else [sample_id]
            writer.writerow(row)


def train_model(epochs=Config.NUM_EPOCHS, load_cached=True):
    """
    Main training routine.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data Preparation
    train_loader, val_loader, test_loader = get_dataloaders()

    # Load Ground Truth for Validation
    val_df = pd.read_csv(Config.VAL_CSV)
    gt_map = {}
    for _, row in val_df.iterrows():
        lbls = row["labels"]
        if pd.isna(lbls) or lbls == "":
            gt_map[row["sample_id"]] = []
        else:
            # Handle potential float/int mix if read by pandas weirdly
            gt_map[row["sample_id"]] = [int(float(x)) for x in str(lbls).split(",")]

    # 2. Model Setup
    model = BAMPNet().to(device)

    # Loss Setup
    # Class weights: 0.5 for background (0), 1.0 for others
    class_weights = torch.ones(Config.NUM_CLASSES + 1).to(device)
    class_weights[Config.BACKGROUND_CLASS_ID] = 0.5

    criterion_class = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    best_ler = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_class, None, device
        )

        # Validate
        val_ler = validate(model, val_loader, gt_map, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Val LER: {val_ler:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_ler < best_ler:
            best_ler = val_ler
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # 4. Submission
    print("Loading best model for submission...")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
