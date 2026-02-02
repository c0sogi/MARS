import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_dataset, ChatbotArenaDataset
from library.model import SiameseDebertaModel
from library.engine import train_one_epoch, validate, predict


def run_demo():
    print("==== Starting Library Demonstration ====")

    # 1. Configuration Setup for Demo
    # We override paths to keep the demo isolated and fast
    print("[1] Configuring environment...")
    Config.OUTPUT_DIR = "./working/demo_run"
    Config.CACHE_DIR = "./working/demo_run/cache"
    Config.TRAIN_CACHE_FILE = os.path.join(Config.CACHE_DIR, "train_data.parquet")
    Config.VAL_CACHE_FILE = os.path.join(Config.CACHE_DIR, "val_data.parquet")
    Config.TEST_CACHE_FILE = os.path.join(Config.CACHE_DIR, "test_data.parquet")

    # Reduce hyperparameters for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.GRADIENT_ACCUMULATION_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Create directories
    Config.setup()
    seed_everything(Config.SEED)
    print(f"    Output Directory: {Config.OUTPUT_DIR}")

    # 2. Data Loading
    print("\n[2] Loading and Processing Data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load a tiny subset (limit=16) to ensure speed
    # We set load_cached_data=False to verify the processing logic
    train_dataset = load_dataset("train", tokenizer, load_cached_data=False, limit=16)
    val_dataset = load_dataset("val", tokenizer, load_cached_data=False, limit=16)

    print(f"    Train Dataset Size: {len(train_dataset)}")
    print(f"    Val Dataset Size: {len(val_dataset)}")

    # Verify Dataset Item Structure
    sample_item = train_dataset[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "response_mask",
        "scalars",
        "target",
    ]
    for key in required_keys:
        assert key in sample_item, f"Missing key {key} in dataset item"

    # Check shapes
    # input_ids should be [2, seq_len] (Branch A and B)
    assert sample_item["input_ids"].shape[0] == 2, "Input IDs should have 2 branches"
    assert sample_item["scalars"].shape[0] == 3, "Scalars should have 3 features"
    assert sample_item["target"].shape[0] == 3, "Target should have 3 classes"
    print("    Dataset structure verified.")

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = SiameseDebertaModel()
    model.to(device)
    print("    Model moved to device.")

    # Verify Forward Pass
    print("    Verifying forward pass...")
    # Create a dummy batch
    collate_fn = None  # Default collate is fine for stacked tensors
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"].to(device)
    scalars = batch["scalars"].to(device)

    with torch.no_grad():
        # Enable autocast to match engine logic
        with torch.amp.autocast(device_type="cuda", enabled=Config.USE_FP16):
            logits = model(input_ids, attention_mask, response_mask, scalars)

    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(Config.TRAIN_BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"
    print("    Forward pass successful. Output shape verified.")

    # 4. Training Loop
    print("\n[4] Testing Training Loop...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Simple scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Initialize Scaler
    scaler = torch.amp.GradScaler("cuda")

    avg_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        epoch=1,
    )

    assert isinstance(avg_loss, float), "train_one_epoch should return a float loss"
    print(f"    Training One Epoch Complete. Average Loss: {avg_loss:.4f}")

    # 5. Validation
    print("\n[5] Testing Validation...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    val_loss, val_metrics = validate(model, val_loader, device)

    assert "log_loss" in val_metrics, "Validation metrics missing 'log_loss'"
    print(f"    Validation Complete. Loss: {val_loss:.4f}, Metrics: {val_metrics}")

    # 6. Prediction (Test Mode)
    print("\n[6] Testing Prediction...")
    # We'll use the validation set as a proxy for test set here
    # In real usage, we would load "test" split which has no targets
    test_loader = val_loader

    preds = predict(model, test_loader, device)

    assert preds.shape == (len(val_dataset), 3), "Prediction shape mismatch"
    # Check if probabilities sum to roughly 1
    row_sums = preds.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-5
    ), "Predictions are not valid probabilities"

    print("    Prediction successful.")
    print(f"    Sample Predictions:\n{preds[:3]}")

    print("\n==== Demonstration Complete ====")


if __name__ == "__main__":
    # Ensure we don't suppress errors for the demo
    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL ERROR IN DEMO: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
