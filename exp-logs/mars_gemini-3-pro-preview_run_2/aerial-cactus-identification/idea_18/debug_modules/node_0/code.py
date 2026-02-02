import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.dataset as dataset
import library.model as model_lib
import library.train as train_lib
import library.inference as inference_lib
import library.utils as utils


def main():
    print(">>> Starting Cactus Classification Demo")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override config for a fast demonstration run
    config.DEBUG = True
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 16
    config.WORKING_DIR = "./working/demo_execution"
    config.setup_directories()

    # Set reproducibility
    utils.set_seed(42)
    device = torch.device(config.DEVICE)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n>>> Loading Datasets...")
    # Load datasets (DEBUG=True limits the size to 100 samples)
    # We set load_cached_data=False to demonstrate the loading logic from metadata
    train_ds, val_ds, test_ds = dataset.get_datasets(load_cached_data=False)

    # FIX: The provided train.py's `train_one_epoch` expects unpacking (images, labels).
    # The dataset returns (images, labels, ids) if ids are present.
    # We explicitly remove ids from train/val datasets to ensure compatibility.
    train_ds.ids = None
    val_ds.ids = None

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    # Create DataLoaders
    # We use num_workers=0 to avoid overhead in this short script
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verification
    batch_imgs, batch_lbls = next(iter(train_loader))
    assert batch_imgs.shape == (
        config.BATCH_SIZE,
        3,
        32,
        32,
    ), "Incorrect train batch image shape"
    assert batch_lbls.shape == (config.BATCH_SIZE,), "Incorrect train batch label shape"
    print("Data shapes verified.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n>>> Initializing Model...")
    model = model_lib.WideResNetPyramidal().to(device)

    # Verify architecture output
    dummy_x = torch.randn(2, 3, 32, 32).to(device)
    out_mid, out_final = model(dummy_x)
    assert out_mid.shape == (2, 1) and out_final.shape == (
        2,
        1,
    ), "Model output shape mismatch"
    print("Model architecture verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n>>> Starting Training...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Run training for the configured epochs (1 epoch in this demo)
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_lib.train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = train_lib.validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}"
        )

    # Save the model
    # inference_lib expects the model file to be named 'model_seed_{seed}.pth'
    model_save_path = os.path.join(config.WORKING_DIR, "model_seed_42.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("\n>>> Running Inference...")
    # Load the model using the inference library
    loaded_model = inference_lib.load_model(seed=42, device=device)

    # Predict using Test Time Augmentation (TTA)
    ids, probs = inference_lib.predict_tta(loaded_model, test_loader, device)

    # Verification
    assert len(ids) == len(test_ds), "Prediction count mismatch"
    assert probs.shape == (len(test_ds), 1), "Probability shape mismatch"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"
    print(f"Generated predictions for {len(ids)} test images.")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    print("\n>>> Saving Submission...")
    submission_df = pd.DataFrame({"id": ids, "has_cactus": probs.flatten()})

    sub_path = os.path.join(config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(">>> Demo Complete.")


if __name__ == "__main__":
    main()
