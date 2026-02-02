import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed, StandardScaler
from library.data import get_train_val_datasets, get_test_dataset
from library.model import GCCGCNN
from library.engine import train_one_epoch, evaluate


def main():
    print("Starting GC-CGCNN Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration for Demo Run
    # -------------------------------------------------------------------------
    # Modify Config class attributes to run a fast demonstration
    print("Configuring parameters for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Process only 100 samples
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size

    # Set up a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize directories
    Config.setup()

    # Set random seeds for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Computation device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\nLoading Training and Validation Data...")
    # load_cached=False forces reprocessing of the small debug subset
    train_dataset, val_dataset = get_train_val_datasets(load_cached=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # Verify we have data
    assert len(train_dataset) > 0, "Training dataset is empty!"
    assert len(val_dataset) > 0, "Validation dataset is empty!"

    # Create DataLoaders
    # num_workers=0 is often safer for simple scripts/demos to avoid IPC overhead/errors
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # -------------------------------------------------------------------------
    # 3. Preprocessing (Scaling)
    # -------------------------------------------------------------------------
    print("\nInitializing Target Scaler...")
    # Collect all target values from the training set to fit the scaler
    all_train_targets = []
    for data in train_dataset:
        if data.y is not None:
            all_train_targets.append(data.y)

    # Concatenate into a single tensor [N_train, Num_Targets]
    if all_train_targets:
        targets_tensor = torch.cat(all_train_targets, dim=0)
        scaler = StandardScaler(device=device)
        scaler.fit(targets_tensor)
        print(f"Scaler Mean: {scaler.mean.cpu().numpy()}")
        print(f"Scaler Std:  {scaler.std.cpu().numpy()}")
    else:
        raise RuntimeError("No targets found in training data. Cannot fit scaler.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\nInitializing GCCGCNN Model...")
    model = GCCGCNN(config=Config).to(device)
    print("Model created successfully.")

    # Optimizer and Loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = torch.nn.MSELoss()

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting Training Loop...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # Validate
        val_loss, val_metrics, _ = evaluate(
            model, val_loader, criterion, scaler, device
        )

        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
        print(f"  Train Loss (Scaled MSE): {train_loss:.4f}")
        print(f"  Val Loss (Scaled MSE):   {val_loss:.4f}")
        print(f"  Val MAE (Formation):     {val_metrics.get('mae_formation', 0.0):.4f}")
        print(f"  Val MAE (Bandgap):       {val_metrics.get('mae_bandgap', 0.0):.4f}")

    # Save the model
    print(f"\nSaving model checkpoint to {Config.CHECKPOINT_DIR}...")
    torch.save(
        model.state_dict(),
        os.path.join(Config.CHECKPOINT_DIR, "best_model_runfile.pth"),
    )

    # -------------------------------------------------------------------------
    # 6. Inference on Test Set
    # -------------------------------------------------------------------------
    print("\nLoading Test Data...")
    test_dataset = get_test_dataset(load_cached=False)
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Running Inference...")
    # evaluate returns (loss, metrics, predictions_dict)
    # For test set, loss and metrics might be 0/empty if no targets exist
    _, _, test_results = evaluate(model, test_loader, criterion, scaler, device)

    test_ids = test_results["ids"]
    test_preds = test_results["preds"]

    print(f"Generated predictions for {len(test_ids)} samples.")

    # -------------------------------------------------------------------------
    # 7. Generate Submission
    # -------------------------------------------------------------------------
    print("\nGenerating Submission File...")

    # Ensure we have predictions
    if len(test_ids) == 0:
        raise RuntimeError("No predictions generated for test set.")

    submission_df = pd.DataFrame(
        {
            "id": test_ids,
            "formation_energy_ev_natom": test_preds[:, 0],
            "bandgap_energy_ev": test_preds[:, 1],
        }
    )

    # Sort by ID just in case
    submission_df.sort_values("id", inplace=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print("First 5 rows of submission:")
    print(submission_df.head())

    print("\nDemo script completed successfully.")


if __name__ == "__main__":
    main()
