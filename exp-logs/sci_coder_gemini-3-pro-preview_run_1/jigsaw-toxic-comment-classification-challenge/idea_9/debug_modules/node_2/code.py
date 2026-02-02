import os
import sys
import torch
import torch.nn as nn
import numpy as np
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import prepare_mlm_loaders, prepare_kfold_loaders, prepare_test_loader
from library.model import CustomDeberta
from library.awp import AWP
from library.engine import train_mlm, train_fn, valid_fn, inference_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast debug pass.
    print("Configuring for fast debug execution...")
    Config.debug = True
    Config.debug_subset_size = 64  # Small subset for demonstration
    Config.epochs = 1
    Config.mlm_epochs = 1
    Config.batch_size = 4
    Config.mlm_batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for small debug run

    # Ensure working directory is clean for this run
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    seed_everything(Config.seed)
    device = Config.device
    print(f"Device: {device}")

    # 2. Data Loading: MLM
    print("\n--- Step 1: Preparing MLM Data Loaders ---")
    # This function loads text from train and test, tokenizes, and returns a loader
    mlm_loader = prepare_mlm_loaders(debug=Config.debug, load_cached_data=False)

    # Verify MLM Batch
    batch = next(iter(mlm_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    print(f"MLM Batch loaded. Input shape: {batch['input_ids'].shape}")

    # 3. Engine: MLM Training
    print("\n--- Step 2: Running MLM Pre-training (Domain Adaptation) ---")
    # This trains the backbone on the unlabeled text and saves it to Config.mlm_model_dir
    train_mlm(mlm_loader, device, epochs=Config.mlm_epochs)

    # Verify model was saved
    assert os.path.exists(os.path.join(Config.mlm_model_dir, "config.json"))
    assert os.path.exists(
        os.path.join(Config.mlm_model_dir, "model.safetensors")
    ) or os.path.exists(os.path.join(Config.mlm_model_dir, "pytorch_model.bin"))
    print("MLM training complete and model saved.")

    # 4. Data Loading: Supervised Training (Fold 0)
    print("\n--- Step 3: Preparing Supervised Data Loaders (Fold 0) ---")
    train_loader, val_loader = prepare_kfold_loaders(
        fold=0, debug=Config.debug, load_cached_data=False
    )

    # Verify Supervised Batch
    batch = next(iter(train_loader))
    assert "labels" in batch
    assert batch["labels"].shape[1] == Config.num_labels
    print(f"Train Batch loaded. Labels shape: {batch['labels'].shape}")

    # 5. Model Initialization
    print("\n--- Step 4: Initializing Custom Model ---")
    # We initialize the model. We can optionally load the MLM weights we just trained.
    # For demonstration, we point it to the directory where train_mlm saved the weights.
    model = CustomDeberta(pretrained=True, checkpoint_path=Config.mlm_model_dir)
    model.to(device)

    # Verify Forward Pass logic manually
    print("Verifying forward pass...")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    output = model(input_ids, attention_mask, labels)
    assert "logits" in output
    assert "loss" in output
    assert output["logits"].shape == (Config.batch_size, Config.num_labels)
    print(f"Forward pass successful. Loss: {output['loss'].item():.4f}")

    # 6. Supervised Training with AWP
    print("\n--- Step 5: Running Supervised Training with AWP ---")

    # Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
    )

    # Initialize AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=0,  # Force AWP to start immediately for this demo
    )

    # Train for 1 epoch
    avg_loss = train_fn(
        train_loader, model, optimizer, scheduler, device, epoch=0, awp=awp
    )
    print(f"Epoch 0 Training Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # 7. Validation
    print("\n--- Step 6: Running Validation ---")
    val_loss, val_score = valid_fn(val_loader, model, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation ROC AUC: {val_score:.4f}")
    assert 0.0 <= val_score <= 1.0, "ROC AUC score out of bounds"

    # 8. Inference
    print("\n--- Step 7: Running Inference on Test Set ---")
    test_loader, test_ids = prepare_test_loader(
        debug=Config.debug, load_cached_data=False
    )

    predictions = inference_fn(test_loader, model, device)

    print(f"Predictions shape: {predictions.shape}")
    assert predictions.shape == (len(test_ids), Config.num_labels)
    assert (
        predictions.min() >= 0.0 and predictions.max() <= 1.0
    ), "Predictions must be probabilities"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
