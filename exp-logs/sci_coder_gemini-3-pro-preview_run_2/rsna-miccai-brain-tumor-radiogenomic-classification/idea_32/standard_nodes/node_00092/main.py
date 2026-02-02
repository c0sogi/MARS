import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided library modules
from library import config, utils, data_loader, model as lib_model


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Initialization
    # -------------------------------------------------------------------------
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # Load cached data to speed up initialization
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model & Optimizer
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    net = lib_model.AsymmetricEfficientNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop (Fast Baseline)
    # -------------------------------------------------------------------------
    # We use config.NUM_EPOCHS (15) to allow sufficient convergence while relying on early stopping.
    n_epochs = config.NUM_EPOCHS
    best_auc = 0.0

    print(f"Starting training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        # Train
        train_loss = lib_model.train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = lib_model.validate(net, val_loader, criterion, device)

        # Save Best
        if val_auc > best_auc:
            best_auc = val_auc
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                checkpoint_dir=config.CACHE_DIR,
            )

    # -------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # -------------------------------------------------------------------------
    print("Loading best model for final assessment...")
    best_path = os.path.join(config.CACHE_DIR, "best_model.pth")
    if os.path.exists(best_path):
        utils.load_checkpoint(net, path=best_path, device=device)

    net.eval()
    val_probs = []
    val_targets = []

    # Run inference on validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            val_probs.extend(probs)
            val_targets.extend(targets.numpy().flatten())

    val_auc_final = roc_auc_score(val_targets, val_probs)
    print(f"Final Validation Metric: {val_auc_final}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")

    # Load validation metadata to correlate errors with features
    val_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Ensure length matches (val_loader is not shuffled, so order is preserved)
    if len(val_df) == len(val_probs):
        val_df["prob"] = val_probs
        val_df["target"] = val_targets
        val_df["error"] = np.abs(val_df["target"] - val_df["prob"])

        # Extract Slice Counts for each modality to see if depth affects error
        # (Re-implementing lightweight feature extraction from EDA)
        modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
        print("Correlating Prediction Error with Slice Counts:")

        for mod in modalities:
            counts = []
            for _, row in val_df.iterrows():
                path_col = f"path_{mod}"
                full_path = os.path.join(config.INPUT_DIR, row[path_col])
                try:
                    # Fast count of files
                    n_slices = len(
                        [
                            name
                            for name in os.listdir(full_path)
                            if os.path.isfile(os.path.join(full_path, name))
                        ]
                    )
                except Exception:
                    n_slices = 0
                counts.append(n_slices)

            val_df[f"{mod}_slices"] = counts

            # Calculate correlation
            if len(val_df) > 1:
                corr, _ = pearsonr(val_df["error"], val_df[f"{mod}_slices"])
                print(f"Correlation (Error vs {mod}_slices): {corr}")
    else:
        print(
            "Warning: Mismatch between validation dataframe and prediction length. Skipping detailed failure analysis."
        )

    # -------------------------------------------------------------------------
    # 7. Conditional Submission
    # -------------------------------------------------------------------------
    threshold = 0.6321818181818182

    if val_auc_final > threshold:
        print(
            f"\nValidation metric {val_auc_final} > {threshold}. Generating submission..."
        )
        lib_model.predict_and_submit(net, test_loader, device, config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {val_auc_final} <= {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
