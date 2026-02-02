import os
import sys
import torch
import warnings
import numpy as np
import pandas as pd
import transformers
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.dataset import process_data, get_dataloaders, get_test_dataloader
from library.model import CustomModel
from library.loss import HybridLoss
from library.training_utils import get_optimizer_params, AWP, EMA
from library.engine import train_fn, valid_fn, inference_fn


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    # 1. Setup and Configuration
    # We override defaults to ensure the demo runs fast (debug mode, tiny model, 1 epoch)
    print("Initializing Configuration...")
    cfg = Config(
        debug=True,
        debug_sample_size=50,  # Small subset for speed
        epochs=1,
        train_batch_size=4,
        valid_batch_size=4,
        model_name="prajjwal1/bert-tiny",  # Lightweight model for demonstration
        working_dir="./working/demo_run",
        output_dir="./working/demo_run/output",
        print_freq=5,
        awp_start_epoch=0,  # Force AWP to run immediately for demo
        ema_start_epoch=0,
    )

    set_seed(cfg.seed)
    device = cfg.device
    print(f"Device: {device}")

    # Suppress verbose warnings
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()

    # 2. Data Preparation
    print("\n--- Data Preparation ---")
    # This will load metadata, merge CPC context, and create folds
    # It caches to parquet, so we test that flow too.
    df_processed = process_data(cfg, load_cached_data=False)
    print(f"Processed DataFrame shape: {df_processed.shape}")

    # Get DataLoaders for Fold 0
    train_loader, val_loader = get_dataloaders(cfg, fold=0, load_cached_data=True)
    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    expected_keys = {"input_ids", "attention_mask", "id", "labels"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Missing keys in batch. Found: {batch.keys()}"
    assert batch["input_ids"].shape == (
        cfg.train_batch_size,
        cfg.max_len,
    ), "Incorrect input_ids shape"
    assert batch["labels"].shape == (cfg.train_batch_size,), "Incorrect labels shape"
    print("Data batch structure verified.")

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    model = CustomModel(cfg)
    model.to(device)

    # Verify Forward Pass
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, mask)

    assert "logits" in outputs, "Model output missing 'logits'"
    assert "class_logits" in outputs, "Model output missing 'class_logits'"
    assert outputs["logits"].shape == (
        cfg.train_batch_size,
        1,
    ), "Regression logits shape mismatch"
    assert outputs["class_logits"].shape == (
        cfg.train_batch_size,
        5,
    ), "Classification logits shape mismatch"
    print("Model forward pass verified.")

    # 4. Loss Function
    print("\n--- Loss Function ---")
    criterion = HybridLoss(cfg)
    labels = batch["labels"].to(device)

    loss_dict = criterion(outputs, labels)
    assert "loss" in loss_dict, "Loss dict missing 'loss'"
    assert "pearson" in loss_dict, "Loss dict missing 'pearson'"
    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN"
    print(f"Computed Loss: {loss_dict['loss'].item():.4f}")

    # 5. Training Setup
    print("\n--- Training Setup ---")
    # Optimizer with LLRD
    optimizer_params = get_optimizer_params(model, cfg)
    optimizer = AdamW(optimizer_params, lr=cfg.encoder_lr, eps=cfg.eps)

    # Scheduler
    num_training_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # AWP & EMA
    awp = AWP(
        model,
        optimizer,
        adv_lr=cfg.awp_lr,
        adv_eps=cfg.awp_eps,
        start_epoch=cfg.awp_start_epoch,
    )
    ema = EMA(model, decay=cfg.ema_decay)

    # 6. Training Loop Execution
    print("\n--- Running Training Loop (1 Epoch) ---")
    avg_loss = train_fn(
        fold=0,
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        awp=awp,
        ema=ema,
    )
    print(f"Epoch 0 Train Loss: {avg_loss:.4f}")

    # 7. Validation Execution
    print("\n--- Running Validation ---")
    val_loss, val_pearson, val_preds = valid_fn(
        fold=0,
        valid_loader=val_loader,
        model=model,
        criterion=criterion,
        device=device,
        cfg=cfg,
        ema=ema,
    )
    print(f"Validation Pearson: {val_pearson:.4f}")
    assert len(val_preds) == len(val_loader.dataset), "Prediction count mismatch"

    # 8. Inference
    print("\n--- Running Inference ---")
    test_loader = get_test_dataloader(cfg, load_cached_data=True)
    ids, predictions = inference_fn(test_loader, model, device, cfg, ema=ema)

    assert len(ids) == len(predictions), "ID and Prediction count mismatch"
    print(f"Generated {len(predictions)} predictions.")

    # 9. Submission Generation
    print("\n--- Generating Submission ---")
    submission_df = pd.DataFrame({"id": ids, "score": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(cfg.submission_path), exist_ok=True)
    submission_df.to_csv(cfg.submission_path, index=False)
    print(f"Submission saved to {cfg.submission_path}")

    # Verify Submission
    saved_df = pd.read_csv(cfg.submission_path)
    assert list(saved_df.columns) == ["id", "score"], "Submission columns mismatch"
    assert len(saved_df) == len(test_loader.dataset), "Submission row count mismatch"
    print("Submission verification successful.")


if __name__ == "__main__":
    run_demo()
