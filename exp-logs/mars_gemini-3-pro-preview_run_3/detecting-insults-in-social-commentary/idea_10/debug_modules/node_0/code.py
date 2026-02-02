import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, logging as transformers_logging
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import load_processed_data, get_dataloader
from library.models import CustomModel
from library.engine import train_teacher_fn, train_student_awp_fn, predict_fn
from library.awp import AWP

# Suppress verbose transformer warnings
transformers_logging.set_verbosity_error()


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Configuration Setup
    # We use debug=True to limit data size (100 rows) and epochs for speed.
    print("\n[1] Initializing Configuration...")
    config = Config(debug=True, epochs=1, train_batch_size=8)

    # Override paths and settings for the demo
    config.working_dir = "./working/demo_run"
    config.model_names = ["roberta-base"]  # Use base model for speed
    config.teacher_epochs = 1
    config.student_epochs = 1
    config.awp_start_epoch = 0  # Force AWP to run immediately

    # Create working directories
    os.makedirs(config.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)

    # 2. Data Loading & Verification
    print("\n[2] Loading and Verifying Data...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_names[0])

    # Create DataLoaders
    # Note: debug=True in Config restricts data to first 100 rows
    train_loader = get_dataloader(config, "train", tokenizer, shuffle=True)
    val_loader = get_dataloader(config, "val", tokenizer, shuffle=False)

    # Verify Train Loader
    batch = next(iter(train_loader))
    assert "input_ids" in batch, "Batch missing input_ids"
    assert "labels" in batch, "Batch missing labels"
    assert batch["input_ids"].shape[0] <= config.train_batch_size, "Batch size mismatch"
    print(f"    Train batch shape verified: {batch['input_ids'].shape}")

    # 3. Model Initialization & Verification
    print("\n[3] Initializing Model...")
    device = config.device
    model = CustomModel(config.model_names[0], config)
    model.to(device)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    with torch.no_grad():
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)

    assert logits.shape == (ids.shape[0],), f"Logits shape mismatch: {logits.shape}"
    print("    Forward pass successful.")

    # 4. Teacher Training
    print("\n[4] Running Teacher Training...")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * config.teacher_epochs,
    )
    teacher_save_path = os.path.join(config.working_dir, "teacher_demo.bin")

    teacher_model, best_auc = train_teacher_fn(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        config,
        teacher_save_path,
    )
    print(f"    Teacher training complete. Best Val AUC: {best_auc:.4f}")

    # 5. Soft Target Generation (Knowledge Distillation Prep)
    print("\n[5] Generating Soft Targets for Distillation...")
    # Generate predictions on validation set (acting as unlabeled data for demo purposes)
    teacher_probs = predict_fn(teacher_model, val_loader, device)

    assert len(teacher_probs) == len(val_loader.dataset), "Prediction count mismatch"
    assert np.all(
        (teacher_probs >= 0) & (teacher_probs <= 1)
    ), "Probabilities out of range"
    print(f"    Generated {len(teacher_probs)} soft targets.")

    # Create Distillation DataLoader
    # We pass the generated probabilities as soft_targets
    distill_loader = get_dataloader(
        config, "val", tokenizer, shuffle=True, soft_targets=teacher_probs
    )

    # Verify Distillation Batch
    d_batch = next(iter(distill_loader))
    assert "soft_targets" in d_batch, "Distillation batch missing soft_targets"
    assert d_batch["soft_targets"].shape == (
        d_batch["input_ids"].shape[0],
    ), "Soft target shape mismatch"
    print("    Distillation DataLoader verified.")

    # 6. Student Training with AWP
    print("\n[6] Running Student Training with AWP...")
    # Initialize a fresh student model
    student_model = CustomModel(config.model_names[0], config)
    student_model.to(device)

    optimizer_student = AdamW(student_model.parameters(), lr=config.learning_rate)
    scheduler_student = get_linear_schedule_with_warmup(
        optimizer_student,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * config.student_epochs,
    )
    student_save_path = os.path.join(config.working_dir, "student_demo.bin")

    # Train student using both labeled data and soft targets
    student_model, student_auc = train_student_awp_fn(
        student_model,
        train_loader,  # Labeled Data
        distill_loader,  # Unlabeled/Distillation Data
        val_loader,  # Validation Data
        optimizer_student,
        scheduler_student,
        device,
        config,
        student_save_path,
    )
    print(f"    Student training complete. Best Val AUC: {student_auc:.4f}")

    # 7. Explicit AWP Logic Verification
    print("\n[7] Verifying AWP Mechanism Explicitly...")
    # We create a simple dummy model to verify weights are actually perturbed and restored
    dummy_model = torch.nn.Linear(10, 1).to(device)
    awp_handler = AWP(dummy_model, config, adv_param="weight")

    # Create dummy input and target
    dummy_input = torch.randn(2, 10).to(device)
    dummy_target = torch.randn(2, 1).to(device)

    # Forward and Backward to populate gradients
    loss = torch.nn.MSELoss()(dummy_model(dummy_input), dummy_target)
    loss.backward()

    # Save original weights
    original_weight = dummy_model.weight.data.clone()

    # Attack: Perturb weights
    awp_handler.attack()
    perturbed_weight = dummy_model.weight.data.clone()

    # Verify perturbation occurred (assuming non-zero gradients)
    diff = torch.norm(original_weight - perturbed_weight)
    print(f"    Weight perturbation magnitude: {diff.item():.6f}")
    if torch.norm(dummy_model.weight.grad) > 1e-6:
        assert not torch.allclose(
            original_weight, perturbed_weight
        ), "AWP failed to perturb weights"

    # Restore: Revert weights
    awp_handler.restore()
    restored_weight = dummy_model.weight.data

    # Verify restoration
    assert torch.allclose(
        original_weight, restored_weight
    ), "AWP failed to restore original weights"
    print("    AWP mechanism verified successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
