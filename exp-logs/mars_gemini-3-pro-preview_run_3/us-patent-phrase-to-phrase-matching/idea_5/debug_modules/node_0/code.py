import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")


# Mock tqdm to suppress progress bars from the library modules
class MockTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


# Patch tqdm before importing library modules that use it
import tqdm.auto

tqdm.auto.tqdm = MockTqdm

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.cpc_utils import load_context_enriched_data
from library.dataset import (
    get_tokenizer,
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader,
    PhraseDataset,
)
from library.model import CustomModel
from library.loss import HybridLoss
from library.awp import AWP
from library.engine import (
    run_dapt,
    get_optimizer_params,
    train_fn,
    valid_fn,
    inference_fn,
)
from transformers import get_cosine_schedule_with_warmup


def run_demo():
    # =========================================================================
    # 1. Configuration Setup
    # =========================================================================
    print(">>> Setting up configuration...")

    # Override Config for a fast demo run
    Config.debug = True  # Use small subset of data
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.dapt_batch_size = 4
    Config.dapt_epochs = 1
    Config.working_dir = "./working/demo_run"
    Config.output_dir = "./working/demo_run"
    Config.model_save_path = os.path.join(Config.working_dir, "model.pth")
    Config.dapt_model_path = os.path.join(Config.working_dir, "dapt_model")
    Config.awp_start_epoch = 0  # Enable AWP immediately for demo
    Config.print_freq = 10

    # Ensure directories exist
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.seed)

    # Initialize Logger
    logger = get_logger("demo")
    logger.info("Configuration configured for demo run.")

    # =========================================================================
    # 2. Data Loading & Verification
    # =========================================================================
    print("\n>>> Loading and Verifying Data...")

    # Load Tokenizer
    tokenizer = get_tokenizer()

    # Load DataLoaders
    # These functions internally use load_context_enriched_data which handles caching
    train_loader = get_train_dataloader(tokenizer)
    val_loader = get_val_dataloader(tokenizer)
    test_loader, test_ids = get_test_dataloader(tokenizer)

    # Verify Train Batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    bin_labels = batch["bin_labels"]

    print(f"Train Batch Size: {input_ids.shape[0]}")
    print(f"Sequence Length: {input_ids.shape[1]}")

    # Assertions
    assert input_ids.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), f"Expected input_ids shape {(Config.train_batch_size, Config.max_len)}, got {input_ids.shape}"
    assert labels.shape[0] == Config.train_batch_size, "Labels batch size mismatch"
    assert (
        bin_labels.shape[0] == Config.train_batch_size
    ), "Bin labels batch size mismatch"

    logger.info("Data loading and batch structure verified.")

    # =========================================================================
    # 3. Domain Adaptive Pre-training (DAPT) Demo
    # =========================================================================
    print("\n>>> Running DAPT (Minimal Demo)...")
    # This will run on the debug subset (very fast) and save the model
    run_dapt(tokenizer)

    assert os.path.exists(
        Config.dapt_model_path
    ), "DAPT model directory was not created."
    logger.info("DAPT completed successfully.")

    # =========================================================================
    # 4. Model Initialization
    # =========================================================================
    print("\n>>> Initializing Model...")

    device = Config.device
    # Initialize model (optionally loading from DAPT path if we wanted to use it)
    # For demo speed, we just use the config name or the DAPT path if it saved correctly
    model_path = (
        Config.dapt_model_path
        if os.path.exists(Config.dapt_model_path)
        else Config.model_name
    )
    model = CustomModel(config_path=model_path, pretrained=True)
    model.to(device)

    # Verify Forward Pass
    dummy_input = input_ids.to(device)
    dummy_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(dummy_input, dummy_mask)

    logits = outputs["logits"]
    class_logits = outputs["class_logits"]

    print(f"Logits Shape: {logits.shape}")
    print(f"Class Logits Shape: {class_logits.shape}")

    assert logits.shape == (
        Config.train_batch_size,
        1,
    ), "Regression output shape mismatch"
    assert class_logits.shape == (
        Config.train_batch_size,
        Config.hybrid_num_classes,
    ), "Classification output shape mismatch"

    logger.info("Model initialized and forward pass verified.")

    # =========================================================================
    # 5. Training Loop Demo
    # =========================================================================
    print("\n>>> Starting Training Loop...")

    # Loss Function
    loss_fn = HybridLoss()

    # Optimizer with LLRD
    optimizer_grouped_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=Config.eps)

    # Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Adversarial Weight Perturbation
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    )

    # Train for 1 epoch (debug mode has very few batches)
    avg_loss = train_fn(
        train_loader,
        model,
        optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        loss_fn=loss_fn,
        awp=awp,
    )

    print(f"Training Epoch 0 complete. Avg Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss returned NaN"

    # =========================================================================
    # 6. Validation
    # =========================================================================
    print("\n>>> Validating...")

    val_loss, val_score = valid_fn(val_loader, model, device, loss_fn)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Pearson Score: {val_score:.4f}")

    # Save Model (simulated best model save)
    torch.save(model.state_dict(), Config.model_save_path)
    logger.info(f"Model saved to {Config.model_save_path}")

    # =========================================================================
    # 7. Inference & Submission
    # =========================================================================
    print("\n>>> Running Inference...")

    predictions = inference_fn(test_loader, model, device)

    assert len(predictions) == len(
        test_ids
    ), "Mismatch between predictions and test IDs"

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "score": predictions})

    submission_path = os.path.join(Config.working_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(submission.head())

    # Final check
    assert os.path.exists(submission_path), "Submission file was not created"
    logger.info("Demo run completed successfully.")


if __name__ == "__main__":
    run_demo()
