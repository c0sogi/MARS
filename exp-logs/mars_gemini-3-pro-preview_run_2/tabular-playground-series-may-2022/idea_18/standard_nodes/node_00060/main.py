import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from library
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import HybridResFunnel
from library.engine import train_one_epoch, evaluate, predict


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Cite solution_lesson_node_00053: Maintain total optimization budget.
    # Cite solution_lesson_node_00045: Align epochs to utilize final scheduler stage.
    # We use the full epoch count defined in Config (35) to ensure convergence.
    Config.DEBUG = False

    print(f"Running Hybrid ResFunnel on {device}")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    # load_cached_data=True allows skipping preprocessing if already done
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = HybridResFunnel().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # StepLR scheduler as defined in Config
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCELoss()

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_auc = 0.0
    checkpoint_dir = Config.WORKING_DIR

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate on validation set
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Update Learning Rate
        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Save Checkpoint
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc": best_auc,
            },
            is_best,
            checkpoint_dir,
        )

    # --------------------------------------------------------------------------
    # 5. Final Evaluation (Best Model)
    # --------------------------------------------------------------------------
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device=device)
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model not found. Using current model state.")

    # Re-evaluate to ensure we have the exact metric of the loaded model
    final_loss, final_auc = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_cont_features = []
    all_targets = []
    all_preds = []

    # Collect data from validation set
    with torch.no_grad():
        for batch in val_loader:
            # Features (Continuous)
            cont = batch["continuous"].cpu().numpy()

            # Targets
            target = batch["target"].cpu().numpy()

            # Predictions
            cont_dev = batch["continuous"].to(device)
            cat_dev = batch["categorical"].to(device)
            out = model(cont_dev, cat_dev).cpu().numpy().flatten()

            all_cont_features.append(cont)
            all_targets.append(target)
            all_preds.append(out)

    # Concatenate all batches
    all_cont_features = np.concatenate(all_cont_features, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Calculate Correlations between each feature and the error
    # Features are f_00 to f_30, excluding f_27
    feature_names = [f"f_{i:02d}" for i in range(31) if i != 27]
    correlations = []

    for i in range(all_cont_features.shape[1]):
        feat_col = all_cont_features[:, i]
        # Check for constant columns to avoid division by zero in correlation
        if np.std(feat_col) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 7. Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9970005855169476

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )

        # Generate predictions on Test Set
        test_preds = predict(model, test_loader, device)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "target": test_preds})

        # Save to CSV
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
