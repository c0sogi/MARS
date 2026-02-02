import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, decode_predictions, compute_levenshtein
from library.data_loader import get_dataloaders
from library.model import MPCNet


def train_epoch(model, loader, optimizer, criterion, device):
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
        length = batch["length"].to(device)

        # Forward pass
        # Output: (B, T, NumClasses)
        logits = model(skeleton, audio, length)

        # Flatten for CrossEntropyLoss
        # (B * T, NumClasses)
        logits_flat = logits.view(-1, Config.NUM_CLASSES)
        # (B * T)
        labels_flat = labels.view(-1)

        # Compute Loss (Supervised Padding: do not mask)
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability with RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Runs validation and computes Levenshtein Error Rate.
    """
    model.eval()
    total_distance = 0
    total_gt_gestures = 0

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)  # (B, T)
            length = batch["length"].to(device)  # (B,)

            # Forward pass
            logits = model(skeleton, audio, length)
            probs = torch.softmax(logits, dim=2)

            # Convert to CPU numpy for decoding
            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            lengths_np = length.cpu().numpy()

            # Process each sample in the batch
            batch_preds = []
            batch_targets = []

            for i in range(len(probs_np)):
                # 1. Decode Predictions
                # Slice valid frames based on length (though model output is padded)
                # We pass the full padded probs to decode_predictions,
                # but technically we should only consider valid frames for the median filter to be accurate at boundaries.
                # However, decode_predictions applies median filter then RLE.
                # Masking the probs to [0,0...1...0] (Background) for padded areas is implicit
                # since the model learns to predict 0 there.

                # Let's slice valid probabilities to be precise
                valid_len = lengths_np[i]
                sample_probs = probs_np[i, :valid_len, :]

                pred_seq = decode_predictions(sample_probs)
                batch_preds.append(pred_seq)

                # 2. Extract Ground Truth Sequence
                # The dataset provides frame-wise labels. We need the sequence of gestures.
                # We can run a simple RLE on the non-zero labels.
                valid_labels = labels_np[i, :valid_len]

                gt_seq = []
                if len(valid_labels) > 0:
                    # Simple RLE for GT
                    curr = valid_labels[0]
                    if curr != Config.BACKGROUND_CLASS_ID:
                        gt_seq.append(curr)

                    for lbl in valid_labels[1:]:
                        if lbl != curr:
                            curr = lbl
                            if curr != Config.BACKGROUND_CLASS_ID:
                                gt_seq.append(curr)

                batch_targets.append(gt_seq)

            # Compute Levenshtein for this batch
            # We accumulate distance and count manually to calculate the global rate
            for p, t in zip(batch_preds, batch_targets):
                dist = compute_levenshtein(
                    [p], [t]
                )  # returns dist / len(t) if len(t)>0
                # But compute_levenshtein returns rate. We want raw distance here to aggregate correctly.
                # Let's use the nltk logic directly or rely on the utils function returning rate?
                # utils.compute_levenshtein returns total_dist / total_len.
                # So we can just call it on the lists.

                # Re-implement accumulation logic here to be safe with the metric definition
                # Metric: Sum(Distances) / Sum(GT Lengths)
                import nltk

                d = nltk.edit_distance(p, t)
                total_distance += d
                total_gt_gestures += len(t)

    error_rate = total_distance / total_gt_gestures if total_gt_gestures > 0 else 0.0
    return error_rate


def run_training(debug_limit=None):
    """
    Main execution function.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_subset_size=debug_limit
    )

    # 2. Model Initialization
    model = MPCNet().to(device)

    # 3. Optimization Setup
    # Class Weights: 0.5 for Background (0), 1.0 for others
    weights = torch.ones(Config.NUM_CLASSES).to(device)
    weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT_VALUE

    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=Config.LABEL_SMOOTHING, reduction="mean"
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 4. Training Loop
    best_ler = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_ler = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val LER: {val_ler:.10f}"
        )

        # Checkpoint
        if val_ler < best_ler:
            best_ler = val_ler
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! (LER: {best_ler:.10f})")

    print(f"Training complete. Best LER: {best_ler:.10f}")

    # 5. Generate Submission
    print("Generating submission...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # We need to map predictions back to Sample IDs.
    # The collate_fn sorts batches by length descending.
    # We must replicate this sort on the dataframe slice to match IDs.

    # Access the test dataframe directly
    test_df = test_loader.dataset.df
    results = []

    current_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            length = batch["length"].to(device)

            logits = model(skeleton, audio, length)
            probs = torch.softmax(logits, dim=2)

            probs_np = probs.cpu().numpy()
            lengths_np = length.cpu().numpy()
            batch_size = len(probs_np)

            # Get corresponding dataframe slice
            df_slice = test_df.iloc[current_idx : current_idx + batch_size].copy()

            # Sort slice by num_frames descending to match collate_fn behavior
            # Stable sort is preferred
            df_slice["sort_len"] = df_slice["num_frames"]
            df_slice_sorted = df_slice.sort_values(
                by="sort_len", ascending=False, kind="mergesort"
            )

            # Generate predictions and pair with IDs
            for i in range(batch_size):
                valid_len = lengths_np[i]
                sample_probs = probs_np[i, :valid_len, :]
                pred_seq = decode_predictions(sample_probs)

                # Convert list of ints to string
                pred_str = ",".join(map(str, pred_seq))

                # Match with ID
                sample_id = df_slice_sorted.iloc[i]["sample_id"]
                results.append((sample_id, pred_str))

            current_idx += batch_size

    # Save to CSV
    submission_df = pd.DataFrame(results, columns=["Id", "Predicted"])
    # The submission format requires: SessionID,Label1,Label2
    # But wait, looking at the prompt example: "Session00001,2,12,3"
    # It implies a headerless CSV or specific format?
    # "Submission Format: For each sequence... For instance: Session00001,2,12,3"
    # The prompt doesn't explicitly specify a header.
    # However, standard practice usually implies no header or a specific one.
    # Let's write raw lines to be safe and strictly follow the example format.

    with open(Config.SUBMISSION_PATH, "w") as f:
        for sid, pred in results:
            f.write(f"{sid},{pred}\n")

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
