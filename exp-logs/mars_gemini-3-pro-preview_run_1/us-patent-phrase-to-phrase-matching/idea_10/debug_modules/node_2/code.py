import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import logging
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers import logging as hf_logging

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.cpc_mapping import get_cpc_texts
from library.data import (
    get_data_splits,
    preprocess_test_data,
    prepare_loaders,
    prepare_test_loader,
)
from library.model import CustomModel
from library.training_utils import (
    get_optimizer_params,
    train_fn,
    valid_fn,
    inference_fn,
)

# Suppress warnings and verbose logs for cleaner output
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("Initializing Demo Script...")

    # 1. Configuration
    # Enable debug mode to use a small subset of data (100 samples) and fewer folds
    cfg = Config(debug=True)

    # Override working directory for this specific demo to avoid conflicts
    cfg.working_dir = "./working/demo_script_exec"
    cfg.model_dir = os.path.join(cfg.working_dir, "models")
    cfg.predictions_dir = os.path.join(cfg.working_dir, "predictions")
    os.makedirs(cfg.working_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.predictions_dir, exist_ok=True)

    # 2. Seeding
    print(f"Setting seed: {cfg.seed}")
    seed_everything(cfg.seed)

    # 3. Data Processing & CPC Mapping
    print("Demonstrating CPC Mapping and Data Splitting...")

    # Force recompute to demonstrate logic (load_cached_data=False)
    # In a real run, we would likely set this to True.
    cpc_texts = get_cpc_texts(cfg, load_cached_data=False)

    # Validation: Ensure CPC mapping is populated
    assert isinstance(cpc_texts, dict), "CPC texts should be a dictionary"
    assert len(cpc_texts) > 0, "CPC texts dictionary is empty"
    print(f"Successfully loaded {len(cpc_texts)} CPC context descriptions.")

    # Load Train Data with Folds
    train_df = get_data_splits(cfg, load_cached_data=False)

    # Validation: Check DataFrame structure
    assert "fold" in train_df.columns, "Fold column missing in train_df"
    assert "context_text" in train_df.columns, "Context text missing in train_df"
    assert (
        len(train_df) == cfg.debug_sample_size
    ), f"Expected {cfg.debug_sample_size} samples in debug mode"
    print(f"Train data loaded. Shape: {train_df.shape}")

    # Load Test Data
    test_df = preprocess_test_data(cfg, load_cached_data=False)
    assert (
        len(test_df) == cfg.debug_sample_size
    ), "Test data size mismatch in debug mode"
    print(f"Test data loaded. Shape: {test_df.shape}")

    # 4. Tokenizer & DataLoader
    print("Initializing Tokenizer and DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Prepare loaders for Fold 0
    fold = 0
    train_loader, valid_loader = prepare_loaders(fold, train_df, tokenizer, cfg)

    # Validation: Inspect one batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    assert input_ids.shape == (
        cfg.train_batch_size,
        cfg.max_length,
    ), f"Batch shape mismatch. Expected ({cfg.train_batch_size}, {cfg.max_length}), got {input_ids.shape}"
    assert "labels" in batch, "Labels missing in training batch"
    print("DataLoaders created and batch structure verified.")

    # 5. Model Initialization
    print("Initializing Custom Model...")
    model = CustomModel(cfg, pretrained=True)
    model.to(cfg.device)

    # Validation: Forward pass check
    # Move dummy batch to device
    dummy_ids = batch["input_ids"].to(cfg.device)
    dummy_mask = batch["attention_mask"].to(cfg.device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_ids, dummy_mask)

    assert output.shape == (
        cfg.train_batch_size,
    ), f"Model output shape mismatch. Expected ({cfg.train_batch_size},), got {output.shape}"
    print("Model initialized and forward pass verified.")

    # 6. Optimizer & Scheduler
    print("Configuring Optimizer and Scheduler...")
    optimizer_parameters = get_optimizer_params(model, cfg)
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=cfg.learning_rate, eps=cfg.eps, betas=cfg.betas
    )

    num_train_steps = int(len(train_df) / cfg.train_batch_size * cfg.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * cfg.warmup_ratio),
        num_training_steps=num_train_steps,
        num_cycles=cfg.num_cycles,
    )

    # 7. Training Loop Demonstration
    print("Starting Training Loop (1 Epoch for demonstration)...")
    epoch = 0

    # Run training function
    avg_train_loss = train_fn(
        train_loader, model, optimizer, epoch, scheduler, cfg.device, cfg
    )

    assert not np.isnan(avg_train_loss), "Training loss is NaN"
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

    # 8. Validation Loop Demonstration
    print("Starting Validation...")
    avg_val_loss, pearson_score = valid_fn(valid_loader, model, cfg.device, cfg)

    assert not np.isnan(avg_val_loss), "Validation loss is NaN"
    assert -1.0 <= pearson_score <= 1.0, "Pearson score out of range [-1, 1]"
    print(f"Validation Loss: {avg_val_loss:.4f}, Pearson: {pearson_score:.4f}")

    # Save the model (demonstrating artifact saving)
    model_path = os.path.join(cfg.model_dir, f"model_fold_{fold}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 9. Inference Demonstration
    print("Running Inference on Test Set...")
    test_loader = prepare_test_loader(test_df, tokenizer, cfg)
    predictions = inference_fn(test_loader, model, cfg.device)

    # Validation: Predictions
    assert len(predictions) == len(
        test_df
    ), f"Prediction count ({len(predictions)}) does not match test set size ({len(test_df)})"
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions contain values outside [0, 1] range"

    print("Inference completed successfully.")

    # 10. Submission Generation
    print("Generating Submission File...")
    submission = pd.DataFrame({"id": test_df["id"], "score": predictions})

    submission_path = os.path.join(cfg.working_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    print("\nDemo Script Completed Successfully.")


if __name__ == "__main__":
    main()
