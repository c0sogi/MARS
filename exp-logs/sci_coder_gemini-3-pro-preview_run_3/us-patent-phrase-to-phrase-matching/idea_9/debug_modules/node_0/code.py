import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataset
from library.model import PhraseModel
from library.engine import train_fn, valid_fn, EMA


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast demonstration.
    print(">>> Setting up configuration for demo execution...")

    # Ensure reproducibility
    seed_everything(Config.seed)

    # Override Config for speed
    Config.debug = True
    Config.debug_sample_size = 64  # Use a tiny subset
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.print_freq = 5
    Config.working_dir = "./working/demo_execution/"
    Config.awp_start_epoch = 0  # Force AWP to run immediately for demonstration

    # Create working directory
    os.makedirs(Config.working_dir, exist_ok=True)

    logger = get_logger("demo_runner")
    logger.info(f"Device: {Config.device}")

    # 2. Prepare Tokenizer
    # We need the tokenizer to create the dataset
    print("\n>>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 3. Data Loading
    print("\n>>> Preparing Datasets (Debug Mode)...")
    # Load Train and Validation datasets
    # This triggers the caching mechanism in dataset.py
    train_dataset = get_dataset("train", tokenizer, load_cached_data=False)
    val_dataset = get_dataset("val", tokenizer, load_cached_data=False)

    # Verify dataset size
    assert (
        len(train_dataset) == Config.debug_sample_size
    ), f"Expected {Config.debug_sample_size} training samples, got {len(train_dataset)}"
    assert (
        len(val_dataset) == Config.debug_sample_size
    ), f"Expected {Config.debug_sample_size} validation samples, got {len(val_dataset)}"

    # Verify dataset item structure
    sample_item = train_dataset[0]
    required_keys = ["input_ids", "attention_mask", "labels", "id"]
    for key in required_keys:
        assert key in sample_item, f"Dataset item missing key: {key}"

    logger.info(f"Train Dataset Size: {len(train_dataset)}")
    logger.info(f"Val Dataset Size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple debugging
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("\n>>> Initializing Model...")
    model = PhraseModel()
    model.to(Config.device)

    # Verify model forward pass with a dummy batch
    dummy_batch = next(iter(train_loader))
    dummy_input_ids = dummy_batch["input_ids"].to(Config.device)
    dummy_mask = dummy_batch["attention_mask"].to(Config.device)

    with torch.no_grad():
        dummy_output = model(dummy_input_ids, dummy_mask)

    assert "logits" in dummy_output, "Model output missing 'logits'"
    assert "class_logits" in dummy_output, "Model output missing 'class_logits'"
    assert dummy_output["logits"].shape == (
        Config.train_batch_size,
        1,
    ), f"Expected logits shape ({Config.train_batch_size}, 1), got {dummy_output['logits'].shape}"
    assert dummy_output["class_logits"].shape == (
        Config.train_batch_size,
        Config.num_classification_bins,
    ), "Class logits shape mismatch"

    logger.info("Model forward pass verification successful.")

    # 5. Optimizer and Scheduler Setup
    print("\n>>> Setting up Optimizer and Scheduler...")
    # Use the Layer-wise Learning Rate Decay (LLRD) setup from the model class
    optimizer_parameters = model.get_optimizer_params(
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.head_lr,
        weight_decay=Config.weight_decay,
    )

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=Config.learning_rate, eps=1e-6
    )

    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Initialize Exponential Moving Average (EMA)
    ema = None
    if Config.use_ema:
        ema = EMA(model, Config.ema_decay)
        ema.register()
        logger.info("EMA initialized.")

    # 6. Training Loop
    print("\n>>> Starting Training Loop...")

    for epoch in range(Config.epochs):
        # Train
        avg_train_loss = train_fn(
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=Config.device,
            epoch=epoch,
            ema=ema,
        )

        # Validate
        avg_val_loss, pearson_score, preds = valid_fn(
            dataloader=val_loader, model=model, device=Config.device, ema=ema
        )

        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:   {avg_val_loss:.4f}")
        logger.info(f"  Pearson:    {pearson_score:.4f}")

        # Assertions to ensure training is behaving reasonably
        assert not np.isnan(avg_train_loss), "Training loss is NaN"
        assert not np.isnan(avg_val_loss), "Validation loss is NaN"
        assert (
            -1.0 <= pearson_score <= 1.0
        ), f"Pearson score out of range: {pearson_score}"
        assert len(preds) == len(val_dataset), "Prediction count mismatch"

    # 7. Inference Demonstration (on Validation set acting as Test)
    print("\n>>> Generating Submission Format...")

    # In a real scenario, we would load test.csv via get_dataset("test", ...)
    # Here we use the validation predictions generated above.

    # Load the validation metadata to get IDs
    val_df = pd.read_csv(os.path.join(Config.metadata_dir, "val.csv"))
    if Config.debug:
        val_df = val_df.head(Config.debug_sample_size)

    submission_df = pd.DataFrame({"id": val_df["id"], "score": preds})

    # Clip scores to valid range [0, 1] as required by the metric/dataset definition
    submission_df["score"] = submission_df["score"].clip(0, 1)

    print("Sample Submission Output:")
    print(submission_df.head())

    # Save to working directory
    submission_path = os.path.join(Config.working_dir, "submission_demo.csv")
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
