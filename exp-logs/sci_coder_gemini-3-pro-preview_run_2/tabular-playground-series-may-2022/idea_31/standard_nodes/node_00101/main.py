import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import warnings

# Import library components
from library.config import Config
from library.dataset import process_data, ManufacturingDataset
from library.model import PostNormHybridSwiGLU
from library.utils import seed_everything


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    # Load data using library function (handles caching automatically)
    data = process_data(Config, load_cached_data=True)
    (
        X_train_seq,
        X_train_cont,
        y_train,
        X_val_seq,
        X_val_cont,
        y_val,
        X_test_seq,
        X_test_cont,
        test_ids,
    ) = data

    # Create Datasets
    train_ds = ManufacturingDataset(X_train_seq, X_train_cont, y_train)
    val_ds = ManufacturingDataset(X_val_seq, X_val_cont, y_val)

    # Create DataLoaders
    # Pin memory and multiple workers for fast data transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 3. Model & Optimizer Setup
    # --------------------------------------------------------------------------
    model = PostNormHybridSwiGLU(Config).to(device)

    # Configure Strict Decoupled Weight Decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude biases, LayerNorms, and Positional Embeddings from weight decay
        if param.ndim <= 1 or "bias" in name or "norm" in name or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": Config.WD_WEIGHTS},
            {"params": no_decay_params, "weight_decay": Config.WD_BIAS_NORM},
        ],
        lr=Config.LR,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_auc = 0.0
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()

        for seq, cont, target in train_loader:
            seq, cont, target = seq.to(device), cont.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass returns [Heads, Batch] for Multi-Sample Dropout
            logits = model(seq, cont)

            # Calculate MSD Loss (Average loss across heads)
            loss = 0
            for i in range(Config.MSD_HEADS):
                loss += criterion(logits[i], target)
            loss /= Config.MSD_HEADS

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation Step
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for seq, cont, target in val_loader:
                seq, cont = seq.to(device), cont.to(device)
                # Inference returns [Batch] (Dropout disabled)
                logit = model(seq, cont)
                val_preds.append(torch.sigmoid(logit).cpu().numpy())
                val_targets.append(target.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # --------------------------------------------------------------------------
    # 5. Final Evaluation
    # --------------------------------------------------------------------------
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []
    with torch.no_grad():
        for seq, cont, target in val_loader:
            seq, cont = seq.to(device), cont.to(device)
            logit = model(seq, cont)
            val_preds.append(torch.sigmoid(logit).cpu().numpy())
            val_targets.append(target.numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    final_auc = roc_auc_score(val_targets, val_preds)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Calculate correlation between error magnitude and continuous features
    # Features are f_00 to f_30, excluding f_27
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    correlations = []
    # X_val_cont is [Samples, Features]
    for i, col_name in enumerate(cont_cols):
        feat_vals = X_val_cont[:, i]
        # Pearson correlation
        corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations.append((col_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Feature Correlations with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 7. Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9972336610045187

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_ds = ManufacturingDataset(X_test_seq, X_test_cont)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        test_preds = []
        with torch.no_grad():
            for seq, cont in test_loader:
                seq, cont = seq.to(device), cont.to(device)
                logit = model(seq, cont)
                test_preds.append(torch.sigmoid(logit).cpu().numpy())

        test_preds = np.concatenate(test_preds)

        # Save Submission
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission = pd.DataFrame({"id": test_ids, "target": test_preds})
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
