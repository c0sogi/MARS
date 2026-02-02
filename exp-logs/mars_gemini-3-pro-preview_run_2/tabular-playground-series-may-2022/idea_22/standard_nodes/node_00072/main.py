import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import SustainedDepthHybridNet
from library.train import train_one_epoch, validate, predict
from library.utils import seed_everything, compute_auc


def main():
    # 1. Setup
    seed_everything(Config.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True as requested
    print("Loading data...")
    loaders = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    test_ids = loaders["test_ids"]

    # 3. Model Initialization
    model = SustainedDepthHybridNet().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Training Loop
    # Reduced epochs for fast baseline execution
    epochs = 15
    best_auc = 0.0
    save_path = Config.MODEL_SAVE_PATH

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.1e} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)

    print("Training complete.")

    # 6. Final Evaluation & Failure Analysis
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # Collect validation data for analysis
    print("Performing full validation inference...")
    val_probs = []
    val_targets = []
    val_features_list = []

    with torch.no_grad():
        for batch in val_loader:
            cont_data = batch["cont"].to(device)
            seq_data = batch["seq"].to(device)
            targets = batch["target"].to(device)

            logits = model(cont_data, seq_data)
            probs = torch.sigmoid(logits)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_features_list.append(cont_data.cpu().numpy())

    y_pred = np.concatenate(val_probs)
    y_true = np.concatenate(val_targets)
    X_val = np.concatenate(val_features_list)  # Shape: (N, 30)

    # Calculate Metric
    final_metric = compute_auc(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Calculate correlation between features and error
    # X_val has 30 continuous features
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        # Pearson correlation
        feat_col = X_val[:, i]
        if np.std(feat_col) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append((f"f_{i:02d}", corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # 7. Conditional Submission
    THRESHOLD = 0.9970005855169476

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_probs = predict(model, test_loader, device)

        submission_df = pd.DataFrame({"id": test_ids, "target": test_probs})

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
