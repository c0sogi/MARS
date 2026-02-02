import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import HybridNetwork
from library.train import setup_optimizer, train_one_epoch, evaluate, predict

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # --------------------------------------------------------------------------
    # 1. Setup
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # Check for existing processed data in root working dir to save time
    # This prevents reprocessing if the data is already available in the expected format
    if os.path.exists("./working/processed_data.npz"):
        Config.PROCESSED_DATA_PATH = "./working/processed_data.npz"

    # Hyperparameters for Fast Baseline
    # A100 allows for larger batch sizes, speeding up training
    BATCH_SIZE = 2048
    EPOCHS = 3
    DEVICE = Config.DEVICE

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # load_cached_data=True attempts to load from Config.PROCESSED_DATA_PATH
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 3. Model & Optimization
    # --------------------------------------------------------------------------
    model = HybridNetwork().to(DEVICE)
    optimizer = setup_optimizer(model)
    criterion = nn.BCELoss()

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_auc = -1.0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()

    # --------------------------------------------------------------------------
    # 5. Final Evaluation
    # --------------------------------------------------------------------------
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Compute Final Metric on the full validation set
    _, final_val_auc = evaluate(model, val_loader, criterion, DEVICE)
    print(f"Final Validation Metric: {final_val_auc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("Performing failure analysis...")
    model.eval()

    all_targets = []
    all_preds = []
    all_cont_features = []

    # Collect validation data (Features, Targets, Preds)
    # Move to CPU to avoid OOM during accumulation on large datasets
    with torch.no_grad():
        for x_cat, x_cont, targets in val_loader:
            x_cat = x_cat.to(DEVICE)
            x_cont_gpu = x_cont.to(DEVICE)

            preds = model(x_cat, x_cont_gpu)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            all_cont_features.append(x_cont.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_cont_features = np.concatenate(all_cont_features, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Calculate Correlations
    # Continuous features are f_00 to f_30, excluding f_27
    feature_indices = [i for i in range(31) if i != 27]
    feature_names = [f"f_{i:02d}" for i in feature_indices]

    print("Correlation between Error Magnitude and Input Features:")
    for i, name in enumerate(feature_names):
        feat_vals = all_cont_features[:, i]

        # Avoid division by zero if feature is constant
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]

        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 7. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9972883264620234

    if final_val_auc > THRESHOLD:
        # Generate predictions
        test_probs = predict(model, test_loader, DEVICE)

        # Load metadata for IDs to ensure correct alignment
        test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Create DataFrame
        submission_df = pd.DataFrame({"id": test_meta["id"], "target": test_probs})

        # Save to the specified location
        os.makedirs("./submission", exist_ok=True)
        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
