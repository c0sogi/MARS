import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.data import prepare_datasets
from library.model import LANNet
from library.train import Trainer, MaskedL1Loss


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline execution while ensuring convergence.
    # The A100 GPU allows us to process the full dataset quickly.
    # Increasing to 80 epochs to ensure full convergence as per Lesson 39.
    Config.EPOCHS = 80

    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Load data (uses cache if available to speed up startup)
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        load_cached_data=True, debug=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = LANNet(config=Config).to(device)

    # --------------------------------------------------------------------------
    # 4. Training Setup
    # --------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    criterion = MaskedL1Loss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        criterion=criterion,
        patience=10,
    )

    # --------------------------------------------------------------------------
    # 5. Training Loop
    # --------------------------------------------------------------------------
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # --------------------------------------------------------------------------
    # 6. Validation & Metrics
    # --------------------------------------------------------------------------
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_u_outs_list = []
    val_inputs_list = []

    with torch.no_grad():
        for x, y, u_out in val_loader:
            x = x.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            preds = model(x)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(y.cpu())
            val_u_outs_list.append(u_out.cpu())
            val_inputs_list.append(x.cpu())

    val_preds = torch.cat(val_preds_list)
    val_targets = torch.cat(val_targets_list)
    val_u_outs = torch.cat(val_u_outs_list)
    val_inputs = torch.cat(val_inputs_list)

    # Compute Final Metric
    final_metric = compute_metric(val_preds, val_targets, val_u_outs)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 7. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Mask for inspiratory phase (u_out == 0)
    mask = val_u_outs == 0

    # Flatten and mask predictions and targets
    flat_preds = val_preds[mask].numpy()
    flat_targets = val_targets[mask].numpy()
    flat_errors = np.abs(flat_preds - flat_targets)

    # Process features for correlation analysis
    # val_inputs is (N, 80, Features). Reshape to (N*80, Features) then apply mask
    n_features = val_inputs.shape[2]
    flat_inputs = val_inputs.reshape(-1, n_features)[mask.view(-1)].numpy()

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(flat_inputs, columns=Config.FEATURE_COLS)
    analysis_df["error"] = flat_errors

    # Compute correlation
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # --------------------------------------------------------------------------
    # 8. Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.16746872663497925

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_preds_list = []
        with torch.no_grad():
            for x, _, _ in test_loader:
                x = x.to(device)
                preds = model(x)
                test_preds_list.append(preds.cpu().numpy().flatten())

        test_preds = np.concatenate(test_preds_list)

        # Load Test IDs from cache
        if not os.path.exists(Config.TEST_CACHE_IDS):
            raise FileNotFoundError("Test IDs cache not found.")
        test_ids = np.load(Config.TEST_CACHE_IDS).flatten()

        # Create submission directory
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        sub_df = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: test_preds})
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
