import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import load_and_process_data, VentilatorDataset
from library.model import PhysicsLSTM
from library.trainer import Trainer


def main():
    # 1. Configuration & Setup
    # Cite solution_lesson_node_00003: Longer training window
    # Config.EPOCHS is already 50 in config.py, removing override.

    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_X, train_y, val_X, val_y, test_X = load_and_process_data(
        load_cached_data=True
    )

    # Create Datasets
    train_dataset = VentilatorDataset(train_X, train_y, is_test=False)
    val_dataset = VentilatorDataset(val_X, val_y, is_test=False)
    test_dataset = VentilatorDataset(test_X, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Model Training
    print("Initializing model...")
    # Cite solution_lesson_node_00003: Switch to LSTM
    model = PhysicsLSTM(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        bidirectional=Config.BIDIRECTIONAL,
        dropout=Config.DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cite solution_lesson_node_00003: ReduceLROnPlateau scheduler
    # Cite debug_lesson_1: Removed 'verbose' argument as it is deprecated in PyTorch 2.0+
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print("Starting training...")
    trainer = Trainer(model, optimizer, device, scheduler=scheduler)
    trainer.fit(train_loader, val_loader)

    # 4. Validation & Metrics
    print("Loading best model for validation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_inputs_list = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            # Forward pass
            preds = model(x)

            # Store data for analysis (move to CPU)
            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(y.numpy())
            val_inputs_list.append(x.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds_list)
    val_targets = np.concatenate(val_targets_list)
    val_inputs = np.concatenate(val_inputs_list)

    # Calculate Metric (MAE on inspiratory phase)
    # Identify u_out column index
    try:
        u_out_idx = Config.FEATURE_COLS.index("u_out")
    except ValueError:
        raise ValueError("u_out not found in Config.FEATURE_COLS")

    # Generate mask: u_out is normalized, so threshold is 0.0 (mean centered)
    # u_out=0 (inspiratory) -> negative value
    # u_out=1 (expiratory) -> positive value
    u_out_feature = val_inputs[:, :, u_out_idx]
    inspiratory_mask = u_out_feature < 0.0

    # Compute absolute error
    abs_error = np.abs(val_preds - val_targets)

    # Apply mask
    masked_error = abs_error[inspiratory_mask]
    final_metric = masked_error.mean()

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis (Correlation of Error with Features):")
    # Flatten arrays to 1D for correlation calculation, filtering by mask
    flat_mask = inspiratory_mask.flatten()
    flat_error = abs_error.flatten()[flat_mask]

    # Flatten inputs
    flat_inputs = val_inputs.reshape(-1, Config.INPUT_DIM)

    for i, feature_name in enumerate(Config.FEATURE_COLS):
        # Get feature values for inspiratory phase only
        feature_vals = flat_inputs[:, i][flat_mask]

        # Calculate correlation
        if len(feature_vals) > 0 and len(flat_error) > 0:
            corr = np.corrcoef(feature_vals, flat_error)[0, 1]
            print(f"{feature_name}: {corr:.4f}")
        else:
            print(f"{feature_name}: N/A")

    # 6. Submission Generation
    # Cite solution_lesson_node_00003: Check metric before submission
    target_metric = 0.8097341656684875
    if final_metric < target_metric:
        print(f"\nMetric {final_metric} < {target_metric}. Generating submission...")
        test_preds_list = []

        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                preds = model(x)
                test_preds_list.append(preds.cpu().numpy())

        # Concatenate and flatten
        # Shape becomes (N_test_breaths * 80, )
        test_preds_flat = np.concatenate(test_preds_list).flatten()

        # Load test metadata to get IDs
        test_meta = pd.read_csv(Config.TEST_PATH)

        # Ensure lengths match
        if len(test_preds_flat) != len(test_meta):
            print(
                f"Warning: Prediction length {len(test_preds_flat)} does not match metadata length {len(test_meta)}"
            )

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_meta["id"], "pressure": test_preds_flat})

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {final_metric} >= {target_metric}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
