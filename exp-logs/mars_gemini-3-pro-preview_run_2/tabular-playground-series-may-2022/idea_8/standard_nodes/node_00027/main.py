import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders, get_data
from library.model import ResFunnelGLU
from library.train_eval import train_one_epoch, validate


def run():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # Set random seeds for reproducibility
    seed = Config.SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    # Use cached data for speed as per instructions
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Training
    # --------------------------------------------------------------------------
    model = ResFunnelGLU()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Limit epochs for fast baseline execution while ensuring convergence
    epochs = Config.EPOCHS
    best_auc = 0.0
    patience = 0
    best_model_path = Config.MODEL_SAVE_PATH

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_auc = validate(model, val_loader, device)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # --------------------------------------------------------------------------
    # 4. Final Validation Assessment
    # --------------------------------------------------------------------------
    print("Loading best model for final evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model file not found. Using current weights.")

    model.eval()

    # Collect predictions and targets for the full validation set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            logits = model(numeric, categorical)
            probs = torch.sigmoid(logits)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_preds = np.concatenate(val_preds).ravel()
    val_targets = np.concatenate(val_targets).ravel()

    final_auc = roc_auc_score(val_targets, val_preds)
    # Print full precision as required
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming failure analysis...")

    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Retrieve validation features for correlation analysis
    # get_data returns: train_num, train_cat, train_target, val_num, ...
    data_tuple = get_data(load_cached_data=True)
    val_num = data_tuple[3]  # Index 3 is val_num

    # Ensure lengths match (handling potential edge cases with debug slicing)
    if len(errors) != len(val_num):
        val_num = val_num[: len(errors)]

    # Compute correlation between each numerical feature and the error
    correlations = []
    num_features = val_num.shape[1]

    for i in range(num_features):
        feat_col = val_num[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_col) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append((i, corr))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for idx, corr in correlations[:5]:
        print(f"Feature index {idx}: Correlation {corr:.4f}")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    threshold = 0.9957464342157875

    if final_auc > threshold:
        print(
            f"\nValidation metric {final_auc} > {threshold}. Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                numeric = batch["numeric"].to(device)
                categorical = batch["categorical"].to(device)

                logits = model(numeric, categorical)
                probs = torch.sigmoid(logits)
                test_preds.append(probs.cpu().numpy())

        test_preds = np.concatenate(test_preds).ravel()

        submission = pd.DataFrame({"id": test_ids, "target": test_preds})

        sub_path = "./submission/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(f"\nValidation metric {final_auc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    run()
