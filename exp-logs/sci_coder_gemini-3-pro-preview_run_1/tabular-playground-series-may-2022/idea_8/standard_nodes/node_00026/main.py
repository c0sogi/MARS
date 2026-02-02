import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import DeGUTModel
from library.engine import train_one_epoch, evaluate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using the full dataset to ensure high performance (AUC > 0.9966)
    # The A100 GPU can handle the full 35 epochs well within the time limit.
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
    )

    # Determine input dimensions dynamically
    sample_batch = next(iter(train_loader))
    num_feats = sample_batch["num_features"].shape[1]
    vocab_size = len(vocab)
    print(f"Num Features: {num_feats}, Vocab Size: {vocab_size}")

    # 3. Model Initialization
    model = DeGUTModel(num_feats=num_feats, vocab_size=vocab_size)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        anneal_strategy="cos",
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Evaluate on validation set
        val_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val AUC: {val_auc:.10f}")

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model to disk
            torch.save(best_model_state, Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print(f"Training complete. Best Val AUC: {best_auc:.10f}")

    # Load best model for analysis and inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 6. Final Validation Metric
    # Re-evaluate to ensure we print the exact metric of the loaded model
    final_val_auc = evaluate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()

    val_errors = []
    val_feats = []

    # Collect predictions and features from validation set
    with torch.no_grad():
        for batch in val_loader:
            num_features = batch["num_features"].to(device)
            seq_features = batch["seq_features"].to(device)
            target_cls = batch["target_cls"].to(device)

            # Forward pass (inference mode, no masking)
            outputs = model(
                num_features=num_features,
                seq_features=seq_features,
                mask_num=None,
                mask_seq=None,
            )

            logits = outputs["logits_cls"]
            probs = torch.sigmoid(logits).squeeze()

            # Calculate absolute prediction error
            errors = torch.abs(target_cls - probs)

            val_errors.append(errors.cpu().numpy())
            val_feats.append(num_features.cpu().numpy())

    val_errors = np.concatenate(val_errors)
    val_feats = np.concatenate(val_feats)

    # Compute correlation between Error Magnitude and Input Features
    n_features = val_feats.shape[1]
    correlations = []

    for i in range(n_features):
        feat_vals = val_feats[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, val_errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for i, corr in correlations[:5]:
        print(f"Feature index {i}: Correlation {corr:.4f}")

    # 8. Conditional Submission
    THRESHOLD = 0.9966634438643771

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_val_auc} > {THRESHOLD}. Generating submission..."
        )

        all_ids = []
        all_probs = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                num_features = batch["num_features"].to(device)
                seq_features = batch["seq_features"].to(device)
                ids = batch["ids"]

                outputs = model(
                    num_features=num_features,
                    seq_features=seq_features,
                    mask_num=None,
                    mask_seq=None,
                )

                logits = outputs["logits_cls"]
                probs = torch.sigmoid(logits).squeeze()

                # Handle edge case for single-item batch
                if probs.ndim == 0:
                    probs = probs.unsqueeze(0)

                all_probs.extend(probs.cpu().numpy())
                all_ids.extend(ids)

        # Save submission file
        submission_path = Config.SUBMISSION_FILE
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        df_sub = pd.DataFrame({"id": all_ids, "target": all_probs})
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print(df_sub.head())

    else:
        print(
            f"\nValidation metric {final_val_auc} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
