import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.model import RNAModel
from library.utils import mcrmse, get_device, seed_everything
from library.data import get_dataloaders


def loss_fn(y_pred, y_true, config):
    """
    Masked Mean Squared Error.
    Calculates MSE only on the first `config.PRED_LEN` (68) positions.
    """
    # Slice to scored positions
    pred_scored = y_pred[:, : config.PRED_LEN, :]
    true_scored = y_true[:, : config.PRED_LEN, :]

    # Standard MSE
    return nn.MSELoss()(pred_scored, true_scored)


def train_fn(model, loader, optimizer, scheduler, device, config):
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move inputs to device
        inputs = {
            "sequence": batch["sequence"].to(device),
            "loop_type": batch["loop_type"].to(device),
            "pair_dist": batch["pair_dist"].to(device),
        }
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(**inputs)

        # Compute loss
        loss = loss_fn(outputs, targets, config)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for stability of 512-width model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)

        # Optimization step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # Step scheduler after epoch (CosineAnnealing)
    if scheduler is not None:
        scheduler.step()

    return total_loss / num_batches


def eval_fn(model, loader, device, config):
    """
    Evaluates the model on the validation set.
    Computes global MCRMSE.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                "sequence": batch["sequence"].to(device),
                "loop_type": batch["loop_type"].to(device),
                "pair_dist": batch["pair_dist"].to(device),
            }
            targets = batch["targets"].to(device)

            outputs = model(**inputs)

            # Store on CPU to avoid OOM
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute MCRMSE
    score = mcrmse(all_targets, all_preds, num_scored=config.PRED_LEN)

    return score.item()


def predict_and_submit(model, test_loader, device, config):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    # Prepare lists to construct dataframe
    ids_seqpos = []
    preds_reactivity = []
    preds_deg_Mg_pH10 = []
    preds_deg_Mg_50C = []

    # Model predicts 3 channels: reactivity, deg_Mg_pH10, deg_Mg_50C
    # We need to fill deg_pH10 and deg_50C with 0.0

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "sequence": batch["sequence"].to(device),
                "loop_type": batch["loop_type"].to(device),
                "pair_dist": batch["pair_dist"].to(device),
            }
            ids = batch["id"]  # List of strings

            # Forward pass (Full sequence length 107)
            outputs = model(**inputs).cpu().numpy()  # (B, 107, 3)

            # Unpack batch
            for i, sample_id in enumerate(ids):
                sample_preds = outputs[i]  # (107, 3)

                for seq_idx in range(config.SEQ_LEN):
                    # Construct row ID
                    row_id = f"{sample_id}_{seq_idx}"
                    ids_seqpos.append(row_id)

                    # Extract predictions
                    # Channel 0: reactivity
                    # Channel 1: deg_Mg_pH10
                    # Channel 2: deg_Mg_50C
                    preds_reactivity.append(sample_preds[seq_idx, 0])
                    preds_deg_Mg_pH10.append(sample_preds[seq_idx, 1])
                    preds_deg_Mg_50C.append(sample_preds[seq_idx, 2])

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id_seqpos": ids_seqpos,
            "reactivity": preds_reactivity,
            "deg_Mg_pH10": preds_deg_Mg_pH10,
            "deg_pH10": 0.0,  # Not predicted/scored
            "deg_Mg_50C": preds_deg_Mg_50C,
            "deg_50C": 0.0,  # Not predicted/scored
        }
    )

    # Save to file
    submission_path = config.SUBMISSION_PATH
    # Ensure directory exists (though Config usually handles working dir)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_training(debug=False):
    """
    Main execution loop.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    config = Config(debug=debug)
    device = get_device()

    print(f"Device: {device}")
    print(f"Debug Mode: {debug}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 3. Model
    model = RNAModel(config=config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # 5. Training Loop
    best_score = float("inf")

    for epoch in range(config.EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, config)

        # Validate
        val_score = eval_fn(model, val_loader, device, config)

        # Logging
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved! (Score: {best_score:.10f})")

    print(f"Training complete. Best Val MCRMSE: {best_score:.10f}")

    # 6. Inference & Submission
    # Load best weights
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    predict_and_submit(model, test_loader, device, config)
