import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config, NUM_CLASSES
from library.utils import (
    seed_everything,
    get_model_copy,
    save_checkpoint,
    load_checkpoint,
)
from library.data_processing import get_dataloaders
from library.model import DeepSupervisedNet
from library.train import train_one_epoch, validate, generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between prediction error and input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_inputs = []
    all_targets = []
    all_preds = []

    # Collect data
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass (primary head only)
            primary_logits, _ = model(inputs)
            _, predicted = torch.max(primary_logits, 1)

            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    # Concatenate
    X_val = np.concatenate(all_inputs, axis=0)
    y_val = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    # Calculate Error Vector (1 if wrong, 0 if correct)
    errors = (y_val != y_pred).astype(int)

    print(f"Total Validation Samples: {len(errors)}")
    print(f"Total Errors: {errors.sum()}")
    print(f"Error Rate: {errors.mean():.6f}")

    # Calculate Correlation with Features
    # We iterate over columns. X_val is (N, Features)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_col = X_val[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feature_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_col, errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Features correlated with Error:")
    print(f"{'Feature Index':<15} {'Correlation':<15}")
    print("-" * 30)
    for idx, corr in correlations[:10]:
        print(f"{idx:<15} {corr:.6f}")
    print("-" * 30)


def main():
    # 1. Setup
    config = Config()
    # Override epochs for fast baseline execution while maintaining enough capacity to learn
    config.epochs = 20

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True as requested
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    model = DeepSupervisedNet(input_dim, NUM_CLASSES, config)
    model = model.to(device)

    # 4. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
        verbose=True,
    )

    # 5. Training Loop
    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    print(f"Starting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f} - "
            f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            state = {
                "state_dict": get_model_copy(model),
                "best_acc": best_acc,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(state, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered after {patience_counter} epochs.")
                break

    # 6. Final Evaluation
    print("Training complete. Loading best model for final evaluation...")
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device)
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    # Re-run validation to get the exact final metric
    final_loss, final_acc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Conditional Submission
    THRESHOLD = 0.9626291666666666

    if final_acc > THRESHOLD:
        print(
            f"\nValidation metric ({final_acc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, test_ids, device, config)
    else:
        print(
            f"\nValidation metric ({final_acc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
