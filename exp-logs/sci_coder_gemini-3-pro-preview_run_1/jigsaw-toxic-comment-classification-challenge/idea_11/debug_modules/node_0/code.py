import sys
import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

# Ensure the current directory is in the path to import from library
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_score
from library.data import get_dataloaders
from library.model import CustomModel
from library.engine import train_mlm, train_fn, valid_fn, inference_fn


def main():
    # =================================================================
    # 1. Configuration & Setup
    # =================================================================
    print(">>> Initializing Configuration...")
    # Initialize config in debug mode
    config = Config(debug=True)

    # Override specific parameters for a fast demonstration
    config.debug_sample_size = 50  # Use only 50 samples
    config.train_batch_size = 8  # Small batch size
    config.valid_batch_size = 16
    config.dapt_epochs = 1  # 1 epoch for DAPT
    config.teacher_epochs = 1  # 1 epoch for Teacher
    config.awp_start_epoch = 0  # Enable AWP immediately to test logic
    config.num_workers = 0  # Avoid multiprocessing overhead for small data

    # Print modified config
    config.print_config()

    # Set random seeds for reproducibility
    seed_everything(config.seed)

    # Load Tokenizer
    print(f">>> Loading Tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # =================================================================
    # 2. Stage 1: Domain-Adaptive Pre-training (DAPT)
    # =================================================================
    print("\n" + "=" * 40)
    print("Stage 1: DAPT (Masked Language Modeling)")
    print("=" * 40)

    # Get DAPT DataLoader
    dapt_loader = get_dataloaders(config, tokenizer, stage="dapt")

    # Verify DAPT Batch Structure
    dapt_batch = next(iter(dapt_loader))
    assert "input_ids" in dapt_batch
    assert "labels" in dapt_batch  # MLM labels
    print(f"DAPT Batch Loaded. Input Shape: {dapt_batch['input_ids'].shape}")

    # Initialize MLM Model (Standard HF Model for MLM)
    # Note: CustomModel is for classification, so we use AutoModelForMaskedLM here
    dapt_model = AutoModelForMaskedLM.from_pretrained(config.model_name)
    dapt_model.to(config.device)

    dapt_optimizer = AdamW(dapt_model.parameters(), lr=config.dapt_lr)

    # Run DAPT Training Loop
    dapt_loss = train_mlm(
        dapt_model,
        dapt_loader,
        dapt_optimizer,
        scheduler=None,
        device=config.device,
        epoch=0,
        config=config,
    )
    print(f"DAPT Completed. Avg Loss: {dapt_loss:.4f}")

    # Clean up to save memory
    del dapt_model, dapt_optimizer, dapt_loader
    torch.cuda.empty_cache()

    # =================================================================
    # 3. Stage 2: Teacher Training (Supervised)
    # =================================================================
    print("\n" + "=" * 40)
    print("Stage 2: Teacher Training (Supervised)")
    print("=" * 40)

    # Get Teacher DataLoaders
    train_loader, val_loader = get_dataloaders(config, tokenizer, stage="teacher")

    # Verify Teacher Batch Structure
    train_batch = next(iter(train_loader))
    assert train_batch["input_ids"].shape == (config.train_batch_size, config.max_len)
    assert train_batch["labels"].shape == (config.train_batch_size, config.num_labels)
    print("Teacher DataLoaders Verified.")

    # Initialize Custom Classification Model
    model = CustomModel(config, pretrained=True)
    model.to(config.device)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.teacher_lr, weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * config.teacher_epochs,
    )

    # Train Loop (Teacher)
    train_loss = train_fn(
        model, train_loader, optimizer, scheduler, config.device, epoch=0, config=config
    )

    # Validation Loop
    val_loss, val_preds = valid_fn(model, val_loader, config.device, config)

    # Calculate Score
    # Extract ground truth labels from the validation dataset
    val_labels = val_loader.dataset.labels

    # Ensure dimensions match
    assert val_preds.shape == val_labels.shape

    auc_score = get_score(val_labels, val_preds)
    print(f"Teacher Validation ROC AUC: {auc_score:.4f}")

    # =================================================================
    # 4. Inference & Pseudo-Labeling
    # =================================================================
    print("\n" + "=" * 40)
    print("Inference & Pseudo-Label Generation")
    print("=" * 40)

    # Get Inference DataLoader
    inference_loader, test_ids = get_dataloaders(config, tokenizer, stage="inference")

    # Run Inference
    test_preds = inference_fn(model, inference_loader, config.device)

    # Verify Inference Output
    assert len(test_preds) == len(test_ids)
    assert test_preds.shape[1] == config.num_labels

    # Create Pseudo-Labels DataFrame
    pseudo_df = pd.DataFrame(test_preds, columns=config.target_cols)
    pseudo_df["id"] = test_ids
    print(f"Generated Pseudo-Labels for {len(pseudo_df)} samples.")

    # =================================================================
    # 5. Stage 3: Student Training (Semi-Supervised)
    # =================================================================
    print("\n" + "=" * 40)
    print("Stage 3: Student Training Setup")
    print("=" * 40)

    # Get Student DataLoaders (Train + Pseudo-Labeled Test)
    student_train_loader, student_val_loader = get_dataloaders(
        config, tokenizer, stage="student", pseudo_labels_df=pseudo_df
    )

    # Verify Student Batch
    student_batch = next(iter(student_train_loader))
    assert student_batch["input_ids"].shape[0] == config.train_batch_size
    # Labels should be float (soft labels)
    assert student_batch["labels"].dtype == torch.float32

    print("Student DataLoaders initialized successfully.")
    print("Student training loop is identical to Teacher loop (skipped for brevity).")

    # =================================================================
    # 6. Submission File Generation
    # =================================================================
    print("\n" + "=" * 40)
    print("Generating Submission File")
    print("=" * 40)

    submission = pd.DataFrame(test_preds, columns=config.target_cols)
    submission.insert(0, "id", test_ids)

    # Save to working directory
    sub_path = os.path.join(config.output_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to: {sub_path}")

    # Verify file exists
    assert os.path.exists(sub_path)

    print("\n>>> All demonstrations completed successfully!")


if __name__ == "__main__":
    main()
