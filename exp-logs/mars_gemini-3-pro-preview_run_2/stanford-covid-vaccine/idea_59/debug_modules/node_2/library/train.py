import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    SEQ_LEN,
    NUM_TARGETS,
    BATCH_SIZE,
    LR,
    EPOCHS,
    SEED,
    set_seed,
)
from library.loss_metric import MCRMSELoss, GlobalMCRMSE
from library.data import get_loaders, Preprocessor
from library.model import DSRDN

# Ensure reproducibility
set_seed(SEED)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using the recycling mechanism.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets, pairs in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        pairs = pairs.to(device)

        B, _, L = inputs.shape

        optimizer.zero_grad()

        # --- Recycling Mechanism ---

        # 1. Static Branch (Computed once)
        z = model.forward_static(inputs)

        # 2. Pass 1: Initial feedback is zero
        y_prev_0 = torch.zeros((B, NUM_TARGETS, L), device=device)
        e_fb_1 = model.forward_feedback(y_prev_0)
        preds_1 = model.forward_interaction(z, e_fb_1, pairs)

        # 3. Pass 2: Feedback is detached predictions from Pass 1
        # Detach to stop gradients flowing back through the feedback loop generation itself
        # (We treat the feedback as a fixed input for the second pass)
        y_prev_1 = preds_1.detach()
        e_fb_2 = model.forward_feedback(y_prev_1)
        preds_2 = model.forward_interaction(z, e_fb_2, pairs)

        # --- Loss Calculation ---
        # Pass both predictions to the loss function which handles the weighting
        # Loss = MCRMSE(preds_2) + 0.5 * MCRMSE(preds_1)
        loss = criterion([preds_2, preds_1], targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Global MCRMSE.
    """
    model.eval()
    metric = GlobalMCRMSE()

    with torch.no_grad():
        for inputs, targets, pairs in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            pairs = pairs.to(device)

            B, _, L = inputs.shape

            # --- Recycling Mechanism (Inference) ---

            # 1. Static Branch
            z = model.forward_static(inputs)

            # 2. Pass 1
            y_prev_0 = torch.zeros((B, NUM_TARGETS, L), device=device)
            e_fb_1 = model.forward_feedback(y_prev_0)
            preds_1 = model.forward_interaction(z, e_fb_1, pairs)

            # 3. Pass 2
            # No detach needed in no_grad mode, but logically we use preds_1
            e_fb_2 = model.forward_feedback(preds_1)
            preds_2 = model.forward_interaction(z, e_fb_2, pairs)

            # Update global metric accumulator with final predictions
            metric.update(preds_2, targets)

    return metric.compute()


def train_model(debug=False):
    """
    Main training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Prepare Data
    preprocessor = Preprocessor()
    data = preprocessor.process(load_cached_data=True)
    train_loader, val_loader, _ = get_loaders(data, batch_size=BATCH_SIZE, debug=debug)

    # 2. Initialize Model
    model = DSRDN().to(device)

    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = MCRMSELoss()

    # 4. Training Loop
    best_val_score = float("inf")
    patience_counter = 0
    early_stopping_patience = 5
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation Score: {best_val_score}")
    return best_val_score


def generate_submission(debug=False):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("No model found. Please train first.")
        return

    # 1. Load Data
    preprocessor = Preprocessor()
    data = preprocessor.process(load_cached_data=True)
    _, _, test_loader = get_loaders(data, batch_size=BATCH_SIZE, debug=debug)
    test_ids = data["test"]["ids"]

    # 2. Load Model
    model = DSRDN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    print("Generating predictions...")
    all_preds = []

    with torch.no_grad():
        for inputs, _, pairs in test_loader:
            inputs = inputs.to(device)
            pairs = pairs.to(device)
            B, _, L = inputs.shape

            # --- Recycling Mechanism (Inference) ---
            z = model.forward_static(inputs)

            # Pass 1
            y_prev_0 = torch.zeros((B, NUM_TARGETS, L), device=device)
            e_fb_1 = model.forward_feedback(y_prev_0)
            preds_1 = model.forward_interaction(z, e_fb_1, pairs)

            # Pass 2
            e_fb_2 = model.forward_feedback(preds_1)
            preds_2 = model.forward_interaction(z, e_fb_2, pairs)

            # Store predictions (transpose to [B, L, 5] for easier processing)
            all_preds.append(preds_2.transpose(1, 2).cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)

    # If debug mode, trim IDs to match predictions
    if debug:
        test_ids = test_ids[: len(all_preds)]

    # 3. Format Submission
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_pred = all_preds[i]  # Shape [107, 5]

        for seqpos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_pred[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_vals[col_idx])

            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
