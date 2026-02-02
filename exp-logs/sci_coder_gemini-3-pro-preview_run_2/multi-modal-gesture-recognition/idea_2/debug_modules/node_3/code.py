import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import (
    set_seed,
    setup_logger,
    levenshtein_distance,
    post_process_predictions,
)
from library.dataset import GestureDataset
from library.model import MultiStageTCN
from library.loss import ActionSegmentationLoss
from library.trainer import Trainer

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Overrides for Speed and Demo Isolation
    # We modify Config attributes directly since they are class attributes.
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.NUM_LAYERS = 6  # Reduce model complexity for speed

    # Ensure directories exist
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration set up. Working directory:", Config.WORKING_DIR)

    # 2. Verify Utilities
    print("\n--- Verifying Utilities ---")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [
        1,
        2,
        4,
    ]  # Substitution cost should be 1 (if simple) or dependent on implementation
    # The provided implementation uses cost 1 for sub/del/ins
    dist = levenshtein_distance(seq1, seq2)
    print(f"Levenshtein distance between {seq1} and {seq2}: {dist}")
    assert dist == 1.0, f"Expected distance 1.0, got {dist}"

    # Test Post-Processing
    # Create dummy logits: (Batch=1, Classes=Config.NUM_CLASSES, Time=5)
    # Class 0 is background. Let's simulate: 0, 1, 1, 2, 0
    dummy_logits = torch.zeros(1, Config.NUM_CLASSES, 5)
    # Frame 0: Class 0 (default high)
    dummy_logits[0, 0, 0] = 10.0
    # Frame 1: Class 1
    dummy_logits[0, 1, 1] = 10.0
    # Frame 2: Class 1
    dummy_logits[0, 1, 2] = 10.0
    # Frame 3: Class 2
    dummy_logits[0, 2, 3] = 10.0
    # Frame 4: Class 0
    dummy_logits[0, 0, 4] = 10.0

    # Median window 1 to keep it raw for this simple test
    processed_seqs = post_process_predictions(dummy_logits, median_window=1)
    print(f"Post-processed sequence: {processed_seqs[0]}")
    # Expected: [1, 2] (0s are removed, repeats collapsed)
    assert processed_seqs[0] == [1, 2], f"Expected [1, 2], got {processed_seqs[0]}"
    print("Utilities verification passed.")

    # 3. Dataset and DataLoader
    print("\n--- Verifying Dataset & DataLoader ---")

    # Use a small limit_size to speed up loading
    LIMIT_SAMPLES = 10

    # Initialize Datasets
    # Note: This might trigger processing if cache doesn't exist in the new demo dir
    print(f"Loading Training Dataset (limit={LIMIT_SAMPLES})...")
    train_ds = GestureDataset(
        split="train", load_cached_data=False, limit_size=LIMIT_SAMPLES
    )
    print(f"Loading Validation Dataset (limit={LIMIT_SAMPLES})...")
    val_ds = GestureDataset(
        split="val", load_cached_data=False, limit_size=LIMIT_SAMPLES
    )

    assert len(train_ds) > 0, "Training dataset is empty."
    assert len(val_ds) > 0, "Validation dataset is empty."

    # Create DataLoaders manually
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=GestureDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=GestureDataset.collate_fn,
    )

    # Fetch one batch to verify shapes
    features, targets, mask = next(iter(train_loader))
    print(
        f"Batch Shapes -> Features: {features.shape}, Targets: {targets.shape}, Mask: {mask.shape}"
    )

    # Features: (Batch, Time, InputDim)
    assert features.ndim == 3
    assert features.shape[2] == Config.INPUT_DIM
    # Targets: (Batch, Time)
    assert targets.ndim == 2
    # Mask: (Batch, Time)
    assert mask.ndim == 2
    print("Dataset verification passed.")

    # 4. Model Verification
    print("\n--- Verifying Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStageTCN(
        num_stages=Config.NUM_STAGES,
        num_layers=Config.NUM_LAYERS,
        num_f_maps=Config.NUM_F_MAPS,
        dim=Config.INPUT_DIM,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    features = features.to(device)
    mask = mask.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(features, mask)
    print(f"Model produced {len(outputs)} stage outputs.")
    assert len(outputs) == Config.NUM_STAGES

    last_stage_out = outputs[-1]
    print(f"Last stage output shape: {last_stage_out.shape}")
    # Expected: (Batch, Classes, Time)
    assert last_stage_out.shape[0] == Config.BATCH_SIZE
    assert last_stage_out.shape[1] == Config.NUM_CLASSES
    assert last_stage_out.shape[2] == features.shape[1]
    print("Model verification passed.")

    # 5. Loss Verification
    print("\n--- Verifying Loss ---")
    criterion = ActionSegmentationLoss().to(device)
    loss = criterion(outputs, targets, mask)
    print(f"Computed Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("Loss verification passed.")

    # 6. Trainer & Training Loop
    print("\n--- Verifying Trainer & Training Loop ---")
    logger = setup_logger(log_file=os.path.join(Config.WORKING_DIR, "train.log"))
    trainer = Trainer(logger=logger)

    # Inject our limited loaders into the trainer
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader

    # Run training
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Check if checkpoint exists
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint found at {best_model_path}")
    else:
        # It's possible validation didn't improve if random init was lucky,
        # but usually it saves at least once. If not, we force save for demo.
        print(
            "Checkpoint not found (validation might not have improved). Saving manually for inference demo."
        )
        torch.save(trainer.model.state_dict(), best_model_path)

    # 7. Inference Verification
    print("\n--- Verifying Inference ---")
    # We need to ensure the test dataset uses the limited cache or we limit it manually.
    # The predict method in Trainer instantiates GestureDataset(split='test').
    # We cannot easily inject a limited dataset into predict() without modifying Trainer code
    # or the dataset class logic.
    # However, for the demo, we can manually run the prediction logic using the trainer's model.

    print("Loading Test Dataset (limit=5)...")
    test_ds = GestureDataset(split="test", load_cached_data=False, limit_size=5)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=GestureDataset.collate_fn
    )

    trainer.model.eval()
    all_preds = []

    print("Running inference on limited test set...")
    with torch.no_grad():
        for feats, _, msk in test_loader:
            feats = feats.to(device)
            msk = msk.to(device)

            outs = trainer.model(feats, msk)
            final_out = outs[-1]

            preds = post_process_predictions(
                final_out, median_window=Config.MEDIAN_WINDOW_SIZE
            )
            all_preds.extend(preds)

    print(f"Generated predictions for {len(all_preds)} samples.")
    print(f"Sample prediction: {all_preds[0]}")

    # Save dummy submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    with open(sub_path, "w") as f:
        for i, pred in enumerate(all_preds):
            line = f"SampleTest{i:03d}," + ",".join(map(str, pred))
            f.write(line + "\n")

    print(f"Demo submission saved to {sub_path}")
    assert os.path.exists(sub_path), "Submission file was not created."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
