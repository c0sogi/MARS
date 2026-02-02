import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ToxicityModel
from library.engine import train_fn, eval_fn, predict_fn, set_seed


def run_demonstration():
    print("=== Starting Demonstration Script ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Modify Config class attributes directly to affect all subsequent instantiations
    # We use debug mode to process only 1000 samples for speed
    Config.debug = True
    Config.epochs = 1
    Config.train_batch_size = 8
    Config.valid_batch_size = 16

    # Use a separate working directory for this demo to avoid overwriting
    # or loading the full dataset cache from ./working/idea_3
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    Config.working_dir = demo_working_dir
    Config.model_save_path = os.path.join(demo_working_dir, "model_demo.pth")

    # Set seed for reproducibility
    set_seed(Config.seed)

    # ------------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------------
    print("\n[2] Loading DataLoaders (Debug Mode)...")

    # load_cached_data=True allows using cache if it exists in our new working_dir.
    # Since it's a new dir, it will process the data from scratch (which is fast for 1000 rows).
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verify Train Loader
    print("    Verifying Train Loader batch structure...")
    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # Assertions
    assert input_ids.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), f"Expected input_ids shape {(Config.train_batch_size, Config.max_len)}, got {input_ids.shape}"
    assert attention_mask.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), f"Expected attention_mask shape {(Config.train_batch_size, Config.max_len)}, got {attention_mask.shape}"
    assert labels.shape == (
        Config.train_batch_size,
        Config.num_classes,
    ), f"Expected labels shape {(Config.train_batch_size, Config.num_classes)}, got {labels.shape}"

    print("    Train Loader verification passed.")

    # ------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ------------------------------------------------------------------------
    print("\n[3] Initializing Model...")
    device = Config.device
    model = ToxicityModel()
    model.to(device)

    print("    Verifying Forward Pass...")
    # Move batch to device
    b_input_ids = input_ids.to(device)
    b_attention_mask = attention_mask.to(device)

    # Run forward pass
    with torch.no_grad():
        logits = model(b_input_ids, b_attention_mask)

    # Assert output shape
    assert logits.shape == (
        Config.train_batch_size,
        Config.num_classes,
    ), f"Expected logits shape {(Config.train_batch_size, Config.num_classes)}, got {logits.shape}"

    print("    Forward Pass verification passed.")

    # ------------------------------------------------------------------------
    # 4. Training & Evaluation Loop
    # ------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    # Setup training components
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    total_steps = len(train_loader) * Config.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        total_steps=total_steps,
        pct_start=Config.pct_start,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Train
    train_loss = train_fn(
        model, train_loader, optimizer, scheduler, device, loss_fn, scaler
    )
    print(f"    Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss returned NaN"

    # Evaluate
    print("    Running Evaluation...")
    val_loss, val_auc, col_aucs = eval_fn(model, val_loader, device, loss_fn)

    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation AUC:  {val_auc:.4f}")

    # Assertions
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_auc <= 1.0, f"AUC {val_auc} is out of range [0, 1]"
    assert len(col_aucs) == Config.num_classes, "Column-wise AUCs length mismatch"

    # Save Model (simulating checkpointing)
    torch.save(model.state_dict(), Config.model_save_path)
    print(f"    Model saved to {Config.model_save_path}")

    # ------------------------------------------------------------------------
    # 5. Prediction & Submission
    # ------------------------------------------------------------------------
    print("\n[5] Generating Predictions on Test Set...")

    # Load model state (good practice to verify loading works)
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    test_probs = predict_fn(model, test_loader, device)

    # Verify predictions shape
    # In debug mode, test set is also truncated to 1000 samples
    expected_test_samples = 1000 if Config.debug else 153164
    assert test_probs.shape == (
        expected_test_samples,
        Config.num_classes,
    ), f"Expected predictions shape {(expected_test_samples, Config.num_classes)}, got {test_probs.shape}"

    print("    Creating Submission File...")
    # Load test metadata to get IDs
    test_meta = pd.read_csv(Config.test_metadata_path)

    # If debug mode was used during data loading, the test_loader only has 1000 samples.
    # We need to match the IDs. Since _process_split truncates the dataframe,
    # we should truncate the metadata dataframe similarly for this demo.
    if Config.debug:
        test_meta = test_meta.iloc[:1000]

    submission_df = pd.DataFrame(test_probs, columns=Config().labels)
    submission_df.insert(0, "id", test_meta["id"].values)

    # Save submission
    submission_path = os.path.join(Config.submission_dir, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"    Submission saved to {submission_path}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
