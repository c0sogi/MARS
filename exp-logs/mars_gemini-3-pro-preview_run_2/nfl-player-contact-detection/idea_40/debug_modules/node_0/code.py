import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.features import FeatureEngineering
from library.dataset import ContactDataset
from library.model import LRPNet
from library.engine import LRPNetEngine


def run_demo():
    print("=== Starting LRP-Net Pipeline Demo ===")

    # 1. Configuration Overrides for Speed and Demo Isolation
    print("[1] Configuring environment...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Redirect outputs to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.joblib")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    Config.setup_directories()
    seed_everything(Config.SEED)

    # 2. Feature Engineering
    print("[2] Running Feature Engineering...")
    fe = FeatureEngineering()

    # Load and process Train
    print("    Processing Training Data...")
    X_train, y_train, meta_train = fe.load_and_process_data(
        "train", debug=True, load_cached_data=False
    )

    # Validation assertions for Train
    assert X_train.ndim == 2, "X_train must be 2D"
    assert y_train.ndim == 1, "y_train must be 1D"
    assert len(X_train) == len(y_train), "Mismatch in X and y lengths"
    assert not np.isnan(X_train).any(), "X_train contains NaNs"
    print(f"    Train shape: {X_train.shape}, Positive samples: {y_train.sum()}")

    # Load and process Validation
    print("    Processing Validation Data...")
    X_val, y_val, meta_val = fe.load_and_process_data(
        "validation", debug=True, load_cached_data=False
    )
    assert len(X_val) > 0, "Validation set is empty"
    print(f"    Val shape: {X_val.shape}")

    # 3. Dataset and Dataloader
    print("[3] Creating Datasets and Loaders...")
    train_dataset = ContactDataset(X_train, y_train, training=True)
    val_dataset = ContactDataset(X_val, y_val, training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify DataLoader
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[1] == X_train.shape[1], "DataLoader output dim mismatch"
    assert batch_y.shape[0] == Config.BATCH_SIZE or batch_y.shape[0] == len(
        X_train
    ), "Batch size mismatch"

    # 4. Model Initialization
    print("[4] Initializing LRPNet Model...")
    input_dim = X_train.shape[1]
    model = LRPNet(input_dim=input_dim)

    # Move to device
    device = Config.DEVICE
    model.to(device)

    # Verify Forward Pass
    dummy_input = batch_x.to(device)
    with torch.no_grad():
        dummy_out = model(dummy_input)
    assert dummy_out.shape == (
        dummy_input.size(0),
        1,
    ), f"Model output shape mismatch: {dummy_out.shape}"
    print("    Model initialized and forward pass verified.")

    # 5. Training Loop
    print("[5] Starting Training...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    engine = LRPNetEngine(model, device, optimizer, scheduler)

    # Run fit (1 epoch as configured)
    engine.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training"
    print("    Training complete. Best model saved.")

    # 6. Inference on Test Set
    print("[6] Running Inference on Test Set...")
    # Note: Test set might be small/empty in some environments, but we process what's there
    try:
        X_test, _, meta_test = fe.load_and_process_data(
            "test", debug=True, load_cached_data=False
        )

        if len(X_test) > 0:
            test_dataset = ContactDataset(X_test, training=False)
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            predictions = engine.predict(test_loader)

            assert len(predictions) == len(meta_test), "Prediction count mismatch"

            # 7. Submission Generation
            print("[7] Generating Submission File...")
            submission = meta_test[["contact_id"]].copy()
            submission["contact"] = predictions

            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"    Submission saved to {Config.SUBMISSION_PATH}")
            print(submission.head())
        else:
            print(
                "    Test set empty (likely due to DEBUG sampling filtering all available test plays). Skipping inference."
            )

            # Create dummy submission to satisfy requirements if test set empty
            with open(Config.SUBMISSION_PATH, "w") as f:
                f.write("contact_id,contact\n")

    except Exception as e:
        print(f"    Error during inference/submission: {e}")
        raise e

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
