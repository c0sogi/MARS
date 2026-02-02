import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, MCRMSE, create_submission_file
from library.data import get_dataloaders
from library.model import DeepHierarchicalBiGRU
from library.train import Trainer

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast baseline execution
    Config.NUM_EPOCHS = 10  # Reduced from 20 to ensure quick runtime
    Config.setup()

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    # Load data (utilizing cache if available)
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing Deep Hierarchical BiGRU model...")
    model = DeepHierarchicalBiGRU(
        input_channels=Config.INPUT_CHANNELS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        num_targets=Config.NUM_TARGETS,
    ).to(device)

    # ==========================================
    # 4. Training
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    trainer = Trainer(model, optimizer, scheduler, device, Config)

    print("Starting training...")
    trainer.fit(train_loader, val_loader, Config.NUM_EPOCHS)

    # ==========================================
    # 5. Final Validation & Metric
    # ==========================================
    print("Evaluating best model on full validation set...")

    # Load best model weights
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []

    # Inference on validation set
    with torch.no_grad():
        for features, adjacency, targets in val_loader:
            features = features.to(device)
            adjacency = adjacency.to(device)

            outputs = model(features, adjacency)
            # Use final head output
            final_pred = outputs[-1].cpu()

            val_preds.append(final_pred)
            val_targets.append(targets)

    y_pred = torch.cat(val_preds, dim=0)
    y_true = torch.cat(val_targets, dim=0)

    # Calculate and print strict metric
    final_metric = MCRMSE(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Load validation metadata for feature correlation
    val_meta = pd.read_parquet(Config.VAL_METADATA)

    # Calculate error magnitude per sample (RMSE of scored columns)
    seq_scored = Config.SEQ_SCORED
    scored_cols = Config.get_scored_columns()
    all_targets = Config.get_target_columns()
    scored_indices = [i for i, col in enumerate(all_targets) if col in scored_cols]

    # Slice to relevant data
    y_true_sliced = y_true[:, :seq_scored, scored_indices]
    y_pred_sliced = y_pred[:, :seq_scored, scored_indices]

    # Compute RMSE per sample
    # Shape: (N, Seq, Cols) -> Mean over (Seq, Cols) -> Sqrt
    mse_per_sample = torch.mean(
        (y_true_sliced - y_pred_sliced) ** 2, dim=(1, 2)
    ).numpy()
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Add to metadata
    # Note: We assume val_loader preserves order of val_meta (shuffle=False)
    val_meta["error_magnitude"] = rmse_per_sample

    # Feature Engineering for Analysis
    val_meta["pct_A"] = val_meta["sequence"].apply(lambda s: s.count("A") / len(s))
    val_meta["pct_G"] = val_meta["sequence"].apply(lambda s: s.count("G") / len(s))
    val_meta["pct_C"] = val_meta["sequence"].apply(lambda s: s.count("C") / len(s))
    val_meta["pct_U"] = val_meta["sequence"].apply(lambda s: s.count("U") / len(s))

    analysis_features = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
    ]

    # Calculate correlations
    correlations = val_meta[analysis_features].corrwith(val_meta["error_magnitude"])

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ==========================================
    # 7. Submission
    # ==========================================
    threshold = 0.5884495377540588

    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        test_preds_list = []

        with torch.no_grad():
            for features, adjacency, _ in test_loader:
                features = features.to(device)
                adjacency = adjacency.to(device)

                outputs = model(features, adjacency)
                test_preds_list.append(outputs[-1].cpu().numpy())

        test_preds_arr = np.concatenate(test_preds_list, axis=0)

        create_submission_file(test_ids, test_preds_arr, Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
