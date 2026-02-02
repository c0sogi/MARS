import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, logging

# Import provided library components
from library.config import Config
from library.utils import seed_everything, compute_qwk, get_optimizer_params
from library.dataset import EssayDataset, Collate
from library.model import OrdinalModel
from library.engine import train_loop, generate_submission

# Suppress transformer warnings for cleaner output
logging.set_verbosity_error()

if __name__ == "__main__":
    print("=== Starting Essay Scoring Task Demonstration ===\n")

    # 1. Setup & Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast demo.
    print("1. Configuring environment...")
    seed_everything(Config.seed)

    # Override Config for rapid demonstration
    Config.debug = True  # Uses only 100 samples
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.max_length = 128  # Reduced sequence length for speed
    Config.output_dir = "./working/demo"
    Config.model_save_path = os.path.join(Config.output_dir, "model_demo.pth")
    Config.submission_path = os.path.join(Config.output_dir, "submission.csv")

    # Create demo output directory
    os.makedirs(Config.output_dir, exist_ok=True)

    # Define cache paths for demo to avoid conflicts with full training
    demo_train_cache = os.path.join(Config.output_dir, "demo_train.parquet")
    demo_val_cache = os.path.join(Config.output_dir, "demo_val.parquet")
    demo_test_cache = os.path.join(Config.output_dir, "demo_test.parquet")

    print(f"   Debug Mode: {Config.debug}")
    print(f"   Device: {Config.device}")
    print(f"   Output Directory: {Config.output_dir}\n")

    # 2. Data Preparation
    print("2. Preparing Datasets...")

    # Instantiate Training Dataset
    train_ds = EssayDataset(
        data_path=Config.train_path,
        processed_path=demo_train_cache,
        load_cached_data=False,  # Force reload for demo
        is_test=False,
        debug=Config.debug,
    )

    # Instantiate Validation Dataset
    val_ds = EssayDataset(
        data_path=Config.val_path,
        processed_path=demo_val_cache,
        load_cached_data=False,
        is_test=False,
        debug=Config.debug,
        tokenizer=train_ds.tokenizer,  # Reuse tokenizer
    )

    print(f"   Train Dataset Size: {len(train_ds)}")
    print(f"   Val Dataset Size: {len(val_ds)}")

    # Verify Data Loading Logic
    sample = train_ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert "essay_id" in sample

    # Verify Ordinal Encoding Logic
    # If score is 3, labels should be [1, 1, 0, 0, 0] (Thresholds: >1, >2, >3, >4, >5)
    # We can't know the score of index 0 easily without looking at df, but we can check shape/type.
    assert sample["labels"].shape[0] == Config.num_labels
    assert sample["labels"].dtype == torch.float32
    print("   Dataset verification passed.")

    # Create DataLoaders
    collate_fn = Collate(train_ds.tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Verify Batch Shapes
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape[0] == Config.train_batch_size
    assert "labels" in batch
    print("   DataLoader verification passed.\n")

    # 3. Model Initialization
    print("3. Initializing Model...")
    model = OrdinalModel(
        model_name=Config.model_name, num_labels=Config.num_labels, pretrained=True
    )
    model.to(Config.device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_ids = batch["input_ids"].to(Config.device)
        dummy_mask = batch["attention_mask"].to(Config.device)
        outputs = model(dummy_ids, dummy_mask)

        # Output shape should be [batch_size, num_labels]
        assert outputs.shape == (Config.train_batch_size, Config.num_labels)
        print(f"   Model forward pass successful. Output shape: {outputs.shape}\n")

    # 4. Training Setup
    print("4. Setting up Training Loop...")

    # Optimizer with LLRD
    optimizer_grouped_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate * 5,  # Higher LR for head
        weight_decay=Config.weight_decay,
        llrd_decay=Config.llrd_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Scheduler
    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Run Training
    print("   Starting training (1 epoch)...")
    model = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.device,
        epochs=Config.epochs,
        save_path=Config.model_save_path,
        patience=1,
    )
    print("   Training loop execution finished.\n")

    # 5. Inference & Submission
    print("5. Generating Submission...")

    test_ds = EssayDataset(
        data_path=Config.test_path,
        processed_path=demo_test_cache,
        load_cached_data=False,
        is_test=True,
        debug=Config.debug,  # Use subset for demo speed
        tokenizer=train_ds.tokenizer,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    generate_submission(
        model=model,
        test_loader=test_loader,
        device=Config.device,
        output_path=Config.submission_path,
    )

    # Verify Submission File
    if os.path.exists(Config.submission_path):
        df_sub = pd.read_csv(Config.submission_path)
        print(f"   Submission file created. Shape: {df_sub.shape}")

        # Check columns
        assert "essay_id" in df_sub.columns
        assert "score" in df_sub.columns

        # Check score range
        assert df_sub["score"].min() >= 1
        assert df_sub["score"].max() <= 6
        print("   Submission file format verification passed.\n")
    else:
        raise FileNotFoundError("Submission file was not created.")

    # 6. Metric Verification
    print("6. Verifying Metric Calculation...")
    # Test Case: Perfect agreement
    y_true = [1, 2, 3, 4, 5, 6]
    y_pred = [1, 2, 3, 4, 5, 6]
    qwk = compute_qwk(y_true, y_pred)
    assert np.isclose(qwk, 1.0), f"Expected QWK 1.0, got {qwk}"

    # Test Case: Complete disagreement
    y_true_bad = [1, 1, 1]
    y_pred_bad = [6, 6, 6]
    qwk_bad = compute_qwk(y_true_bad, y_pred_bad)
    # Kappa can be 0 or negative
    print(f"   Metric check passed. Perfect Score: {qwk:.4f}, Bad Score: {qwk_bad:.4f}")

    print("\n=== Demonstration Completed Successfully ===")
