import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dapt_loaders, get_teacher_loaders, get_test_loader
from library.model import CustomModel
from library.engine import run_mlm, train_fn, valid_fn, inference_fn


def main():
    print("Starting Demo Script...")

    # ==========================================
    # 1. Configuration Overrides for Demo
    # ==========================================
    print("Configuring environment for fast execution...")

    # Enable debug mode to use small data subsets
    Config.debug = True
    Config.debug_sample_size = 100  # Small sample for speed

    # Reduce training parameters
    Config.n_folds = 2
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.dapt_batch_size = 4
    Config.dapt_epochs = 1
    Config.teacher_epochs = 1
    Config.print_freq = 5

    # Setup isolated working directory for this demo
    Config.working_dir = "./working/demo_execution"
    Config.output_dir = os.path.join(Config.working_dir, "output")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.dapt_model_path = os.path.join(Config.working_dir, "dapt_backbone")

    # Clean up any previous demo run
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)

    # Create necessary directories
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.cache_dir, exist_ok=True)
    os.makedirs(Config.dapt_model_path, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.seed)

    # ==========================================
    # 2. Tokenizer & Data Loading
    # ==========================================
    print("\n=== Initializing Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    print("\n=== Testing DAPT Data Loading ===")
    # load_cached_data=False ensures we generate the cache for this specific debug run
    dapt_loader = get_dapt_loaders(tokenizer, load_cached_data=False)

    # Verify DAPT batch
    dapt_batch = next(iter(dapt_loader))
    print(f"DAPT Batch Keys: {dapt_batch.keys()}")
    assert "input_ids" in dapt_batch
    assert "labels" in dapt_batch  # DataCollatorForLanguageModeling adds labels
    assert dapt_batch["input_ids"].shape[0] == Config.dapt_batch_size
    print("DAPT Data Loading Verified.")

    print("\n=== Testing Supervised Data Loading (Fold 0) ===")
    train_loader, val_loader = get_teacher_loaders(
        fold=0, tokenizer=tokenizer, load_cached_data=True
    )

    # Verify Train Batch
    train_batch = next(iter(train_loader))
    print(f"Train Batch Keys: {train_batch.keys()}")
    assert "input_ids" in train_batch
    assert "labels" in train_batch
    assert train_batch["labels"].shape[1] == Config.num_labels
    print("Supervised Data Loading Verified.")

    # ==========================================
    # 3. Stage 1: Domain-Adaptive Pre-training (DAPT)
    # ==========================================
    print("\n=== Running DAPT (Masked Language Modeling) ===")
    # Initialize MLM model
    mlm_model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    mlm_model.to(Config.device)

    # Optimizer for DAPT
    mlm_optimizer = AdamW(
        mlm_model.parameters(), lr=Config.dapt_lr, weight_decay=Config.weight_decay
    )

    # Run DAPT
    run_mlm(dapt_loader, mlm_model, mlm_optimizer, Config.device, Config)

    # Verify Model Saving
    assert os.path.exists(os.path.join(Config.dapt_model_path, "config.json"))
    assert os.path.exists(
        os.path.join(Config.dapt_model_path, "model.safetensors")
    ) or os.path.exists(os.path.join(Config.dapt_model_path, "pytorch_model.bin"))
    print("DAPT Completed and Model Saved.")

    # Free memory
    del mlm_model, mlm_optimizer
    torch.cuda.empty_cache()

    # ==========================================
    # 4. Stage 2: Supervised Training (Teacher)
    # ==========================================
    print("\n=== Running Supervised Training (Teacher) ===")
    # Initialize Custom Model
    model = CustomModel(pretrained=True)
    model.to(Config.device)

    # Setup Training Components
    optimizer = AdamW(
        model.parameters(), lr=Config.teacher_lr, weight_decay=Config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * Config.teacher_epochs,
    )

    # Train for 1 epoch
    print("Training...")
    avg_loss = train_fn(
        train_loader, model, criterion, optimizer, 0, scheduler, Config.device, Config
    )
    print(f"Training Loss: {avg_loss:.4f}")

    # Validate
    print("Validating...")
    val_loss, val_preds = valid_fn(val_loader, model, criterion, Config.device, Config)
    print(f"Validation Loss: {val_loss:.4f}")

    # Check Validation Metric
    # Get true labels from validation loader
    val_labels = []
    for batch in val_loader:
        val_labels.append(batch["labels"].numpy())
    val_labels = np.concatenate(val_labels)

    # Calculate AUC (handle potential single-class batches in debug mode safely)
    try:
        score = roc_auc_score(val_labels, val_preds, average="micro")
        print(f"Validation Micro AUC: {score:.4f}")
    except ValueError:
        print(
            "Skipping AUC calculation due to insufficient class diversity in debug sample."
        )

    print("Supervised Training Verified.")

    # ==========================================
    # 5. Stage 3: Inference
    # ==========================================
    print("\n=== Running Inference ===")
    test_loader = get_test_loader(tokenizer, load_cached_data=True)

    test_preds = inference_fn(test_loader, model, Config.device)

    # Verify Output Shape
    # In debug mode, test set is also sampled to debug_sample_size
    expected_rows = Config.debug_sample_size
    assert test_preds.shape == (
        expected_rows,
        Config.num_labels,
    ), f"Expected shape ({expected_rows}, {Config.num_labels}), got {test_preds.shape}"

    print(f"Inference Predictions Shape: {test_preds.shape}")

    # Create Submission File
    submission_df = pd.DataFrame(test_preds, columns=Config.target_cols)
    # We need IDs to make it a valid submission, get them from the loader's dataset df
    submission_df["id"] = test_loader.dataset.df["id"].values

    # Reorder columns to match submission format
    cols = ["id"] + Config.target_cols
    submission_df = submission_df[cols]

    submission_path = os.path.join(Config.working_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
