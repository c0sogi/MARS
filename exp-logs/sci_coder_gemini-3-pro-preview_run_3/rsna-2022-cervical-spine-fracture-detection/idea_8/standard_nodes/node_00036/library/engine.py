import torch
import numpy as np
import pandas as pd
import os
import sys
from library.config import Config
from library.utils import get_weighted_log_loss_score
from library.loss import WeightedMILLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()

    total_loss_sum = 0.0
    num_batches = 0

    for batch_idx, (images, targets, box_targets, has_box, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        box_targets = box_targets.to(device)
        has_box = has_box.to(device)

        optimizer.zero_grad()

        # Forward pass: (B, Seq, 7)
        instance_logits = model(images)

        # Calculate Loss
        loss, metrics = criterion(instance_logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss_sum += metrics["loss"]
        num_batches += 1

    avg_loss = total_loss_sum / num_batches

    return avg_loss, 0.0, 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and the competition metric score.
    """
    model.eval()

    total_loss_sum = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, box_targets, has_box, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            box_targets = box_targets.to(device)
            has_box = has_box.to(device)

            # Forward pass
            instance_logits = model(images)

            # Calculate validation loss (for tracking)
            loss, _ = criterion(instance_logits, targets)
            total_loss_sum += loss.item()
            num_batches += 1

            # Aggregate predictions for metric calculation
            # 1. Global Max Pooling over sequence: (B, S, 7) -> (B, 7)
            pooled_logits, _ = torch.max(instance_logits, dim=1)

            # 2. Derive patient_overall logit: max(C1..C7) -> (B, 1)
            patient_logit, _ = torch.max(pooled_logits, dim=1, keepdim=True)

            # 3. Concatenate: (B, 8)
            global_logits = torch.cat([pooled_logits, patient_logit], dim=1)

            # 4. Sigmoid to get probabilities
            probs = torch.sigmoid(global_logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss_sum / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # Calculate competition metric
    score = get_weighted_log_loss_score(y_pred, y_true)

    return avg_loss, score


def fit(
    model, train_loader, val_loader, optimizer, device, epochs=Config.EPOCHS, patience=3
):
    """
    Main training loop with early stopping and scheduling.
    """
    criterion = WeightedMILLoss().to(device)

    # Scheduler: Decoupled Cosine Annealing
    # T_max set relative to epochs as per config
    t_max = int(epochs * Config.T_MAX_MULT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=Config.MIN_LR
    )

    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, epochs + 1):
        # Train
        t_loss, t_mil, t_box = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        v_loss, v_score = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{epochs} | LR: {current_lr:.8f}")
        print(f"Train Loss: {t_loss:.8f} (MIL: {t_mil:.8f}, Box: {t_box:.8f})")
        print(f"Val Loss:   {v_loss:.8f} | Val Score: {v_score:.8f}")

        # Early Stopping
        if v_score < best_score:
            print(
                f"Score improved from {best_score:.8f} to {v_score:.8f}. Saving model..."
            )
            best_score = v_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Score: {best_score:.8f}")


def inference(model, test_loader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Starting inference...")
    model.eval()

    # Dictionary to store predictions: study_id -> {label: prob}
    study_preds = {}

    # Column names mapping for the 7 vertebrae
    c_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    with torch.no_grad():
        for images, _, _, _, study_ids in test_loader:
            images = images.to(device)

            # Forward pass
            instance_logits = model(images)

            # Aggregate
            pooled_logits, _ = torch.max(instance_logits, dim=1)
            probs = torch.sigmoid(pooled_logits)  # (B, 7)

            probs_np = probs.cpu().numpy()

            for i, study_id in enumerate(study_ids):
                p_c = probs_np[i]  # (7,)
                p_overall = np.max(p_c)  # Derived patient_overall

                preds_dict = {}
                for idx, col in enumerate(c_cols):
                    preds_dict[col] = float(p_c[idx])
                preds_dict["patient_overall"] = float(p_overall)

                study_preds[study_id] = preds_dict

    # Generate Submission File
    # We read sample_submission to get the exact row_ids required
    sample_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    submission_rows = []

    for _, row in sample_df.iterrows():
        row_id = row["row_id"]
        # row_id format: [StudyInstanceUID]_[Target]
        # Example: 1.2.826.0.1.3680043.10001_C1

        # Split by last underscore to separate ID and Type
        parts = row_id.rsplit("_", 1)
        study_id = parts[0]
        prediction_type = parts[1]

        prob = 0.5  # Default fallback

        if study_id in study_preds:
            if prediction_type in study_preds[study_id]:
                prob = study_preds[study_id][prediction_type]

        submission_rows.append({"row_id": row_id, "fractured": prob})

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
