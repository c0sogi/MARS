import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import gc
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloader, load_processed_data
from library.models import CustomModel
from library.engine import (
    train_teacher_fn,
    train_student_awp_fn,
    validate_fn,
    predict_fn,
)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # 1. Configuration
    # We set epochs to 3 to ensure the runtime for training 12 models fits comfortably within the limit.
    config = Config(epochs=3, train_batch_size=8, accumulation_steps=4)

    # Create directories
    os.makedirs(config.teacher_dir, exist_ok=True)
    os.makedirs(config.student_dir, exist_ok=True)

    print("Configuration:")
    print(config.to_dict())

    device = config.device
    print(f"Device: {device}")

    # 2. Data Preparation
    # Pre-load tokenizers
    tokenizers = {}
    for name in config.model_names:
        tokenizers[name] = AutoTokenizer.from_pretrained(name)

    # ==========================================
    # Stage 1: Teacher Ensemble Training
    # ==========================================
    print("\n" + "=" * 40)
    print("Stage 1: Teacher Ensemble Training")
    print("=" * 40)

    teacher_preds_test = []

    for model_name in config.model_names:
        for seed in config.seeds:
            print(f"\nTraining Teacher | Model: {model_name} | Seed: {seed}")
            seed_everything(seed)

            # Loaders
            tokenizer = tokenizers[model_name]
            train_loader = get_dataloader(
                config, "train", tokenizer, shuffle=True, drop_last=True
            )
            val_loader = get_dataloader(config, "val", tokenizer, shuffle=False)
            test_loader = get_dataloader(config, "test", tokenizer, shuffle=False)

            # Model
            model = CustomModel(model_name, config)
            model.to(device)

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            # Scheduler
            num_training_steps = (
                len(train_loader) * config.teacher_epochs // config.accumulation_steps
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.warmup_ratio,
                total_iters=num_training_steps,
            )

            # Train
            save_path = os.path.join(
                config.teacher_dir, f"{model_name.replace('/', '_')}_seed_{seed}.bin"
            )
            model, best_auc = train_teacher_fn(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                config,
                save_path,
            )
            print(f"Best Teacher AUC: {best_auc:.5f}")

            # Predict on Test (for Soft Targets)
            preds = predict_fn(model, test_loader, device)
            teacher_preds_test.append(preds)

            # Cleanup to save memory
            del model, optimizer, scheduler, train_loader, val_loader, test_loader
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate Teacher Predictions (Soft Targets)
    avg_teacher_preds = np.mean(teacher_preds_test, axis=0)
    print(f"\nGenerated Soft Targets for {len(avg_teacher_preds)} test samples.")

    # ==========================================
    # Stage 2: Student Ensemble Training (Distillation + AWP)
    # ==========================================
    print("\n" + "=" * 40)
    print("Stage 2: Student Ensemble Training (Distillation + AWP)")
    print("=" * 40)

    student_preds_val = []
    student_preds_test = []

    # Load Validation Targets for final metric calculation
    df_val = load_processed_data(config, "val")
    val_targets = df_val["Insult"].values.astype(float)

    for model_name in config.model_names:
        for seed in config.seeds:
            print(f"\nTraining Student | Model: {model_name} | Seed: {seed}")
            seed_everything(seed)

            # Loaders
            tokenizer = tokenizers[model_name]

            # Labeled Loader (Train)
            train_loader = get_dataloader(
                config, "train", tokenizer, shuffle=True, drop_last=True
            )

            # Distillation Loader (Test with Soft Targets)
            distill_loader = get_dataloader(
                config,
                "test",
                tokenizer,
                shuffle=True,
                drop_last=True,
                soft_targets=avg_teacher_preds,
            )

            val_loader = get_dataloader(config, "val", tokenizer, shuffle=False)
            test_loader = get_dataloader(config, "test", tokenizer, shuffle=False)

            # Model
            model = CustomModel(model_name, config)
            model.to(device)

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            # Scheduler
            num_training_steps = (
                len(train_loader) * config.student_epochs // config.accumulation_steps
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.warmup_ratio,
                total_iters=num_training_steps,
            )

            # Train with AWP and Distillation
            save_path = os.path.join(
                config.student_dir, f"{model_name.replace('/', '_')}_seed_{seed}.bin"
            )
            model, best_auc = train_student_awp_fn(
                model,
                train_loader,
                distill_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                config,
                save_path,
            )
            print(f"Best Student AUC: {best_auc:.5f}")

            # Predict on Val (for Ensemble Eval)
            val_preds = predict_fn(model, val_loader, device)
            student_preds_val.append(val_preds)

            # Predict on Test (for Submission)
            test_preds = predict_fn(model, test_loader, device)
            student_preds_test.append(test_preds)

            # Cleanup
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                distill_loader,
                val_loader,
                test_loader,
            )
            torch.cuda.empty_cache()

    # ==========================================
    # Evaluation
    # ==========================================
    print("\n" + "=" * 40)
    print("Final Evaluation")
    print("=" * 40)

    # Average Student Predictions on Validation Set
    avg_val_preds = np.mean(student_preds_val, axis=0)

    # Calculate Metric
    final_auc = roc_auc_score(val_targets, avg_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # ==========================================
    # Failure Analysis
    # ==========================================
    print("\nFailure Analysis:")
    # Calculate error
    errors = np.abs(val_targets - avg_val_preds)

    # Calculate lengths for correlation analysis
    df_val["char_len"] = df_val["Comment"].apply(len)
    df_val["word_len"] = df_val["Comment"].apply(lambda x: len(str(x).split()))

    # Correlations
    corr_char, _ = pearsonr(errors, df_val["char_len"])
    corr_word, _ = pearsonr(errors, df_val["word_len"])

    print(f"Correlation between Error and Char Length: {corr_char:.4f}")
    print(f"Correlation between Error and Word Length: {corr_word:.4f}")

    # ==========================================
    # Submission
    # ==========================================
    threshold = 0.9660591133004925
    if final_auc > threshold:
        print(
            f"\nValidation metric {final_auc} > {threshold}. Generating submission..."
        )

        avg_test_preds = np.mean(student_preds_test, axis=0)

        # Load test metadata to construct submission dataframe
        df_test = load_processed_data(config, "test")

        submission = pd.DataFrame()
        submission["Insult"] = avg_test_preds
        submission["Date"] = df_test["Date"]
        submission["Comment"] = df_test["Comment"]

        # Save
        submission.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")
    else:
        print(f"\nValidation metric {final_auc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
