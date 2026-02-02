import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.signal import medfilt

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    MEDIAN_FILTER_KERNEL,
    NUM_CLASSES,
    SEED,
    LOSS_WEIGHT_CLS_BG,
    LOSS_WEIGHT_CLS_FG,
)
from library.utils import set_seed, compute_error_rate
from library.model import SSG_CRCN
from library.loss import CombinedLoss
from library.data_loader import prepare_dataset, GestureDataset, collate_fn


def decode_sequence(frame_labels):
    """
    Decodes a frame-wise label sequence into a list of gesture IDs.
    Collapses repeats and removes background (class 0).
    """
    seq = []
    prev = -1
    for l in frame_labels:
        l = int(l)
        if l != prev:
            if l != 0:  # 0 is background
                seq.append(l)
            prev = l
    return seq


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (feats, lbls, bnds, mask) in enumerate(loader):
        feats = feats.to(device)
        lbls = lbls.to(device)
        bnds = bnds.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Forward pass: returns list of outputs from all stages
        outputs = model(feats, mask)

        # Compute loss
        loss, _ = criterion(outputs, lbls, bnds, mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feats, lbls, bnds, mask in loader:
            feats = feats.to(device)
            lbls = lbls.to(device)
            bnds = bnds.to(device)
            mask = mask.to(device)

            outputs = model(feats, mask)
            loss, _ = criterion(outputs, lbls, bnds, mask)
            total_loss += loss.item()

            # Use final stage output for metrics
            final_stage_out = outputs[-1]  # (B, C+1, T)
            cls_logits = final_stage_out[:, :NUM_CLASSES, :]  # (B, C, T)

            # Get probabilities and argmax
            probs = F.softmax(cls_logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()  # (B, T)

            # Get targets
            targets_np = lbls.cpu().numpy()  # (B, T)
            mask_np = mask.cpu().numpy()  # (B, T)

            # Decode sequences
            for i in range(preds.shape[0]):
                # Extract valid frames based on mask
                valid_len = int(mask_np[i].sum())

                # Raw prediction for this sample
                p_seq = preds[i, :valid_len]
                t_seq = targets_np[i, :valid_len]

                # Apply Median Filter
                if MEDIAN_FILTER_KERNEL > 1:
                    p_seq = medfilt(p_seq, kernel_size=MEDIAN_FILTER_KERNEL)

                # Decode to gesture list
                pred_decoded = decode_sequence(p_seq)
                target_decoded = decode_sequence(t_seq)

                all_preds.append(pred_decoded)
                all_targets.append(target_decoded)

    avg_loss = total_loss / len(loader)
    ler = compute_error_rate(all_preds, all_targets)

    return avg_loss, ler


def train_model():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Prepare Data
    print("Preparing Training Data...")
    train_pos, train_aud, train_lbl, train_bnd, _ = prepare_dataset(
        TRAIN_METADATA_PATH, "train_data", load_cached_data=True
    )

    print("Preparing Validation Data...")
    val_pos, val_aud, val_lbl, val_bnd, _ = prepare_dataset(
        VAL_METADATA_PATH, "val_data", load_cached_data=True
    )

    # Create Datasets
    # Enable augmentation for training
    train_dataset = GestureDataset(
        train_pos, train_aud, train_lbl, train_bnd, augment=True
    )
    val_dataset = GestureDataset(val_pos, val_aud, val_lbl, val_bnd, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = SSG_CRCN().to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_ler = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val LER: {val_ler:.6f}"
        )

        # Early Stopping based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")


def generate_submission():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("No model found. Skipping submission generation.")
        return

    print("Preparing Test Data...")
    test_pos, test_aud, test_lbl, test_bnd, test_ids = prepare_dataset(
        TEST_METADATA_PATH, "test_data", load_cached_data=True
    )

    test_dataset = GestureDataset(test_pos, test_aud, test_lbl, test_bnd, augment=False)
    # Batch size 1 for safe inference sequence generation
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=1
    )

    model = SSG_CRCN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for i, (feats, _, _, mask) in enumerate(test_loader):
            feats = feats.to(device)
            mask = mask.to(device)

            outputs = model(feats, mask)
            final_stage_out = outputs[-1]
            cls_logits = final_stage_out[:, :NUM_CLASSES, :]
            probs = F.softmax(cls_logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()[0]  # (T,)

            # Mask out padding
            valid_len = int(mask[0].sum().item())
            preds = preds[:valid_len]

            # Post-processing
            if MEDIAN_FILTER_KERNEL > 1:
                preds = medfilt(preds, kernel_size=MEDIAN_FILTER_KERNEL)

            decoded_seq = decode_sequence(preds)

            # Format as string
            pred_str = ",".join(map(str, decoded_seq))
            predictions.append(pred_str)

    # Write Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        for sid, pred in zip(test_ids, predictions):
            f.write(f"{sid},{pred}\n")

    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    # Not required by prompt, but good for local testing if run directly
    pass
