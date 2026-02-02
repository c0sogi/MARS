import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from library.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    MODEL_SAVE_PATH,
    ACCUMULATION_STEPS,
    TARGET_COLS,
    SUBMISSION_PATH,
    SEED,
)
from library.utils import seed_everything, WeightedMultiLabelLoss
from library.data import get_dataloaders
from library.model import CervicalSpineSeqModel


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, accumulation_steps
):
    """
    Trains the model for one epoch using gradient accumulation and mixed precision.
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Mixed precision forward pass
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass with scaler
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update running loss (multiply back by accumulation_steps to get actual batch loss)
        running_loss += loss.item() * accumulation_steps

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in enumerate(loader):
            # Unpack tuple if enumerate was used on loader directly,
            # but loader yields (images, targets)
            pass

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            # Standard precision for validation is usually fine,
            # but autocast can be used for consistency/speed
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            running_loss += loss.item()

    return running_loss / len(loader)


def predict(model, loader, device):
    """
    Generates predictions for the entire dataset in the loader.
    Returns:
        np.ndarray: Predictions of shape (N_samples, Num_Classes)
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def generate_submission(model, test_loader, output_path, device):
    """
    Generates the submission CSV file.
    """
    print("Generating predictions for test set...")
    predictions = predict(model, test_loader, device)

    # Retrieve StudyInstanceUIDs from the dataset metadata
    # The test_loader is not shuffled, so order matches the metadata
    test_metadata = test_loader.dataset.metadata
    study_uids = test_metadata["StudyInstanceUID"].values

    if len(study_uids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(study_uids)} studies vs {len(predictions)} predictions."
        )

    submission_rows = []

    # Map predictions to submission format
    # Columns in predictions correspond to TARGET_COLS indices
    for i, uid in enumerate(study_uids):
        probs = predictions[i]
        for class_idx, class_name in enumerate(TARGET_COLS):
            row_id = f"{uid}_{class_name}"
            prob = probs[class_idx]
            submission_rows.append({"row_id": row_id, "fractured": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_model(debug=False):
    """
    Main training pipeline.
    """
    seed_everything(SEED)

    # 1. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # 2. Model
    print("Initializing Model...")
    model = CervicalSpineSeqModel(pretrained=True)
    model.to(DEVICE)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Competition metric is weighted log loss, so we use the custom loss
    # We apply specific class weights and positive class weights to target
    # the 'patient_overall' label and hard fracture cases.
    weights = torch.tensor(config.CLASS_WEIGHTS).to(DEVICE)
    criterion = WeightedMultiLabelLoss(weights=weights, pos_weight=config.POS_WEIGHT)
    criterion.to(DEVICE)

    scaler = GradScaler()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            scaler,
            ACCUMULATION_STEPS,
        )
        val_loss = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    generate_submission(model, test_loader, SUBMISSION_PATH, DEVICE)
