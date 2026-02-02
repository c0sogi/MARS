import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.utils import set_seed, get_device, compute_auc
from library.data import get_dataloaders
from library.model import DualViewHybridResFunnel
from library.train import Trainer, predict


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    WORK_DIR = "./working"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    SUBMISSION_DIR = "./submission"

    # Training Hyperparameters
    # We use 5 epochs to ensure a fast baseline execution while allowing
    # sufficient convergence to potentially meet the high AUC threshold.
    BATCH_SIZE = 1024
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Strict Threshold for Submission Generation
    VAL_AUC_THRESHOLD = 0.9967793385748163

    # Reproducibility and Device
    set_seed(42)
    device = get_device()

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    model_save_path = os.path.join(WORK_DIR, "best_model.pth")

    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n--- Loading Data ---")
    # get_dataloaders handles reading CSVs, processing features, and caching.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=True,
        cache_dir=WORK_DIR,
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    model = DualViewHybridResFunnel(
        num_continuous=30,
        vocab_size=32,
        embedding_dim=32,
        seq_len=10,
        transformer_layers=2,
        backbone_dims=[512, 256, 128],
        dropout=0.35,
    )
    model.to(device)

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("\n--- Starting Training ---")
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # StepLR scheduler to decay learning rate by 0.1 every 2 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    criterion = nn.BCELoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_path=model_save_path,
    )

    trainer.fit(epochs=EPOCHS)

    # --------------------------------------------------------------------------
    # 5. Validation & Metrics
    # --------------------------------------------------------------------------
    print("\n--- Final Validation ---")
    # Load the best model saved during training
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Collect predictions, targets, and inputs for analysis
    val_preds = []
    val_targets = []
    val_inputs_cont = []

    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            preds = model(x_cont, x_cat)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(y.numpy())
            val_inputs_cont.append(x_cont.cpu().numpy())

    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_inputs_cont = np.concatenate(val_inputs_cont, axis=0)

    final_auc = compute_auc(val_targets, val_preds)

    # Print the required metric
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Generate feature names for continuous inputs (f_00 to f_30, excluding f_27)
    feature_names = [f"f_{i:02d}" for i in range(31) if i != 27]

    # Calculate correlation between error magnitude and each feature
    correlations = []
    for idx, feat_name in enumerate(feature_names):
        feat_values = val_inputs_cont[:, idx]

        # Avoid warning/error if feature is constant
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_values)[0, 1]

        correlations.append((feat_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 7. Submission
    # --------------------------------------------------------------------------
    print("\n--- Submission Check ---")
    if final_auc > VAL_AUC_THRESHOLD:
        print(
            f"Validation AUC ({final_auc}) exceeds threshold ({VAL_AUC_THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set
        test_preds = predict(model, test_loader, device)

        # Retrieve test IDs from metadata or raw file
        test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
        if os.path.exists(test_meta_path):
            df_test_meta = pd.read_csv(test_meta_path)
            test_ids = df_test_meta["id"].values
        else:
            df_test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
            test_ids = df_test_raw["id"].values

        # Create submission file
        submission_df = pd.DataFrame({"id": test_ids, "target": test_preds})

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation AUC ({final_auc}) does not exceed threshold ({VAL_AUC_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
