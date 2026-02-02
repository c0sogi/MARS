import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from library
from library.config import Config
from library.dataset import process_data, ManufacturingDataset
from library.model import PostNormHybridSwiGLU
from library.utils import seed_everything


def run_training(debug_mode=False):
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loading
    # Use the library function which handles caching and metadata loading
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

    # Debug mode: slice data for quick check
    if debug_mode:
        print("Debug mode enabled: reducing dataset size.")
        limit = 2048
        X_train_seq, X_train_cont, y_train = (
            X_train_seq[:limit],
            X_train_cont[:limit],
            y_train[:limit],
        )
        X_val_seq, X_val_cont, y_val = (
            X_val_seq[:limit],
            X_val_cont[:limit],
            y_val[:limit],
        )

    # Create Datasets
    train_ds = ManufacturingDataset(X_train_seq, X_train_cont, y_train)
    val_ds = ManufacturingDataset(X_val_seq, X_val_cont, y_val)
    test_ds = ManufacturingDataset(X_test_seq, X_test_cont)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to stabilize batchnorm/stats if any
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = PostNormHybridSwiGLU(Config).to(device)

    # 4. Optimizer with Strict Decoupled Weight Decay
    # Group 1: Decay (Weights)
    # Group 2: No Decay (Biases, Norms, Pos Embeds)
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check for parameters that should not have weight decay
        # 1. Dimensions <= 1 (biases usually)
        # 2. Explicit names: "bias", "norm" (LayerNorm), "pos_embed" (Transformer position)
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

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # --- Train ---
        model.train()
        running_loss = 0.0

        for seq, cont, target in train_loader:
            seq = seq.to(device)
            cont = cont.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Forward pass
            # Training returns [Heads, Batch] due to Multi-Sample Dropout
            logits = model(seq, cont)

            # Calculate MSD Loss (Average loss over heads)
            loss = 0
            for i in range(Config.MSD_HEADS):
                loss += criterion(logits[i], target)
            loss /= Config.MSD_HEADS

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Step scheduler
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for seq, cont, target in val_loader:
                seq = seq.to(device)
                cont = cont.to(device)

                # Inference returns [Batch] (Dropout disabled)
                logit = model(seq, cont)
                pred = torch.sigmoid(logit)

                val_preds.append(pred.cpu().numpy())
                val_targets.append(target.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        val_auc = roc_auc_score(val_targets, val_preds)

        # Print full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc}"
        )

        # --- Early Stopping ---
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
            # print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    # 6. Inference and Submission
    print("Loading best model for inference...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    model.eval()
    test_preds = []

    with torch.no_grad():
        for seq, cont in test_loader:
            seq = seq.to(device)
            cont = cont.to(device)

            logit = model(seq, cont)
            pred = torch.sigmoid(logit)
            test_preds.append(pred.cpu().numpy())

    test_preds = np.concatenate(test_preds)

    # Save Submission
    submission = pd.DataFrame({"id": test_ids, "target": test_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
