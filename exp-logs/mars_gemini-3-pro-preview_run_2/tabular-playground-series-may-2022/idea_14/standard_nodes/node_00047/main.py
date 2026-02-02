import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config, HybridResFunnel, set_seed
from library.dataset import get_dataloaders
from library.engine import train_model, predict_and_submit


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using the full dataset as the A100 GPU is sufficient to process 640k samples quickly (approx 10-15 mins).
    # This ensures we meet the high AUC threshold requirement.
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    model = HybridResFunnel(Config).to(device)

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    # 5. Training
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=Config.EPOCHS,
        patience=5,
        device=device,
        save_path=best_model_path,
    )

    # 6. Validation & Failure Analysis
    print("\n--- Validation & Failure Analysis ---")

    # Load the best model for analysis
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    model.eval()

    val_preds = []
    val_targets = []
    val_features_cont = []

    # Inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            # Handle variable unpacking based on dataset implementation
            if len(batch) == 3:
                x_cat, x_cont, y = batch
            else:
                x_cat, x_cont = batch
                y = None  # Should not happen for val_loader based on dataset.py

            x_cat = x_cat.to(device)
            x_cont_gpu = x_cont.to(device)

            logits = model(x_cat, x_cont_gpu)
            probs = torch.sigmoid(logits)

            val_preds.append(probs.cpu().numpy())
            if y is not None:
                val_targets.append(y.numpy())

            # Keep features on CPU for correlation analysis
            val_features_cont.append(x_cont.numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_features_cont = np.concatenate(val_features_cont, axis=0)

    # Calculate Final Metric
    final_metric = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Features
    errors = np.abs(val_targets.flatten() - val_preds.flatten())

    correlations = []
    # Loop through continuous features (columns of x_cont)
    num_features = val_features_cont.shape[1]
    for i in range(num_features):
        feat_values = val_features_cont[:, i]
        # Calculate Pearson correlation
        if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        else:
            corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 5 Features Correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.6f}")

    # 7. Conditional Submission
    submission_threshold = 0.9970005855169476

    if final_metric > submission_threshold:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({submission_threshold})."
        )
        print("Generating submission...")

        predict_and_submit(
            model=model,
            test_loader=test_loader,
            test_ids=test_ids,
            device=device,
            output_path="./submission/submission.csv",
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({submission_threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
