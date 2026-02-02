import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import get_dataloaders
from library.model import CustomModel
from library.engine import train_mlm, train_fn, valid_fn, inference_fn


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Initialize config
    config = Config(debug=False)

    # Override epochs to ensure execution finishes within 2 hours
    # DAPT: 1 epoch is sufficient to adapt to vocabulary
    # Teacher/Student: 2 epochs each to converge without overfitting given the pre-training
    config.dapt_epochs = 1
    config.teacher_epochs = 2
    config.student_epochs = 2

    # Set seeds
    seed_everything(config.seed)
    print(f"Device: {config.device}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # =========================================================================
    # 2. Stage 1: Domain-Adaptive Pre-training (DAPT)
    # =========================================================================
    print("\n" + "=" * 40)
    print("Stage 1: Domain-Adaptive Pre-training (DAPT)")
    print("=" * 40)

    # Load model for MLM
    dapt_model = AutoModelForMaskedLM.from_pretrained(config.model_name)
    dapt_model.to(config.device)

    # Get DAPT DataLoader (Train + Val + Test text)
    dapt_loader = get_dataloaders(config, tokenizer, stage="dapt")

    # Optimizer
    dapt_optimizer = AdamW(
        dapt_model.parameters(), lr=config.dapt_lr, weight_decay=config.weight_decay
    )

    # Train DAPT
    for epoch in range(config.dapt_epochs):
        train_mlm(
            dapt_model, dapt_loader, dapt_optimizer, None, config.device, epoch, config
        )

    # Save DAPT Backbone
    dapt_save_path = os.path.join(config.working_dir, "dapt_finetuned")
    os.makedirs(dapt_save_path, exist_ok=True)
    dapt_model.save_pretrained(dapt_save_path)
    tokenizer.save_pretrained(dapt_save_path)
    print(f"DAPT model saved to {dapt_save_path}")

    # Cleanup
    del dapt_model, dapt_loader, dapt_optimizer
    torch.cuda.empty_cache()

    # Update config to use the fine-tuned backbone
    config.model_name = dapt_save_path

    # =========================================================================
    # 3. Stage 2: Teacher Training
    # =========================================================================
    print("\n" + "=" * 40)
    print("Stage 2: Teacher Training")
    print("=" * 40)

    # Initialize Teacher Model (Classifier) using DAPT backbone
    teacher_model = CustomModel(config, pretrained=True)
    teacher_model.to(config.device)

    # Get Teacher DataLoaders (Labeled Train / Val)
    train_loader, val_loader = get_dataloaders(config, tokenizer, stage="teacher")

    # Optimizer & Scheduler
    optimizer = AdamW(
        teacher_model.parameters(),
        lr=config.teacher_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.teacher_lr,
        epochs=config.teacher_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.warmup_ratio,
    )

    # Train Teacher
    for epoch in range(config.teacher_epochs):
        train_fn(
            teacher_model,
            train_loader,
            optimizer,
            scheduler,
            config.device,
            epoch,
            config,
        )
        valid_fn(teacher_model, val_loader, config.device, config)

    # =========================================================================
    # 4. Pseudo-Label Generation
    # =========================================================================
    print("\n" + "=" * 40)
    print("Generating Pseudo-Labels for Student Stage")
    print("=" * 40)

    # Get Inference Loader for Test Set
    test_loader, test_ids = get_dataloaders(config, tokenizer, stage="inference")

    # Predict
    teacher_preds = inference_fn(teacher_model, test_loader, config.device)

    # Create DataFrame for pseudo-labels
    pseudo_df = pd.DataFrame(teacher_preds, columns=config.target_cols)
    pseudo_df["id"] = test_ids

    # Cleanup Teacher
    del teacher_model, train_loader, optimizer, scheduler
    torch.cuda.empty_cache()

    # =========================================================================
    # 5. Stage 3: Student Training
    # =========================================================================
    print("\n" + "=" * 40)
    print("Stage 3: Student Training (Semi-Supervised)")
    print("=" * 40)

    # Initialize Student Model from DAPT backbone
    student_model = CustomModel(config, pretrained=True)
    student_model.to(config.device)

    # Get Student DataLoaders (Train + Pseudo-Labeled Test)
    train_loader, val_loader = get_dataloaders(
        config, tokenizer, stage="student", pseudo_labels_df=pseudo_df
    )

    # Optimizer & Scheduler
    optimizer = AdamW(
        student_model.parameters(),
        lr=config.student_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.student_lr,
        epochs=config.student_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.warmup_ratio,
    )

    # Train Student
    best_val_preds = None
    for epoch in range(config.student_epochs):
        train_fn(
            student_model,
            train_loader,
            optimizer,
            scheduler,
            config.device,
            epoch,
            config,
        )
        val_loss, val_preds = valid_fn(student_model, val_loader, config.device, config)
        best_val_preds = val_preds

    # =========================================================================
    # 6. Evaluation & Failure Analysis
    # =========================================================================
    print("\n" + "=" * 40)
    print("Evaluation & Failure Analysis")
    print("=" * 40)

    # Load Ground Truth
    val_df = pd.read_csv(config.val_meta_file)
    y_true = val_df[config.target_cols].values

    # Calculate Metric
    final_score = get_score(y_true, best_val_preds)
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis: Correlation between Error and Input Length
    # Load full validation text from cache
    val_cache_path = os.path.join(config.cache_dir, "val_cache.parquet")
    val_full = pd.read_parquet(val_cache_path)

    # Calculate character length
    val_full["char_len"] = val_full["comment_text"].str.len()

    # Calculate Mean Absolute Error per sample
    error_magnitude = np.mean(np.abs(y_true - best_val_preds), axis=1)

    # Calculate correlation
    correlation = np.corrcoef(val_full["char_len"].values, error_magnitude)[0, 1]
    print(f"Correlation between Error Magnitude and Input Length: {correlation:.4f}")

    # =========================================================================
    # 7. Submission
    # =========================================================================
    threshold = 0.9920879090652149

    if final_score > threshold:
        print(
            f"\nMetric ({final_score}) > Threshold ({threshold}). Generating Submission..."
        )

        # Get Test Loader again
        test_loader, test_ids = get_dataloaders(config, tokenizer, stage="inference")

        # Inference with Student Model
        final_preds = inference_fn(student_model, test_loader, config.device)

        # Format Submission
        sub_df = pd.DataFrame(final_preds, columns=config.target_cols)
        sub_df["id"] = test_ids

        # Ensure column order matches sample submission
        sub_df = sub_df[["id"] + config.target_cols]

        # Save
        sub_df.to_csv(config.submission_file, index=False)
        print(f"Submission saved to {config.submission_file}")
    else:
        print(
            f"\nMetric ({final_score}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
