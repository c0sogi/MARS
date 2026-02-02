import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import InsultDataset, load_and_process_data
from library.model import InsultModel
from library.awp import AWP
from library.engine import train_fn, train_fn_awp, eval_fn


def get_optimizer_params(model, learning_rate, weight_decay):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


def run_training(config, train_df, val_df, test_df, stage="teacher", soft_targets=None):
    logger = get_logger("main")
    device = config.device

    # Prepare validation data (static across all runs)
    # We pick the first tokenizer for validation data preparation to save time,
    # but strictly speaking, we should re-tokenize per model.
    # To be safe and correct, we will re-tokenize inside the loop.
    val_texts = val_df["Comment"].values
    val_targets = val_df["Insult"].values

    # Prepare test data for inference
    test_texts = test_df["Comment"].values

    # Prepare training data based on stage
    if stage == "teacher":
        train_texts = train_df["Comment"].values
        train_targets = train_df["Insult"].values
        logger.info(
            f"Stage 1: Training Teachers on {len(train_texts)} labeled samples."
        )
    else:
        # Stage 2: Combine Labeled Train + Soft Labeled Test
        labeled_texts = train_df["Comment"].values
        labeled_targets = train_df["Insult"].values.astype(float)

        unlabeled_texts = test_df["Comment"].values
        # soft_targets passed in argument

        combined_texts = np.concatenate([labeled_texts, unlabeled_texts])
        combined_targets = np.concatenate([labeled_targets, soft_targets])

        train_texts = combined_texts
        train_targets = combined_targets
        logger.info(
            f"Stage 2: Training Students on {len(train_texts)} combined samples (Labeled + Distilled)."
        )

    # Storage for predictions
    fold_val_preds = []
    fold_test_preds = []

    # Iterate over models and seeds
    # Total models = len(model_names) * len(seeds)
    for model_name in config.model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for seed in config.seeds:
            seed_everything(seed)
            logger.info(f"Training {model_name} [Seed {seed}] - Stage: {stage}")

            # Datasets
            train_dataset = InsultDataset(
                train_texts, tokenizer, config.max_len, train_targets
            )
            val_dataset = InsultDataset(
                val_texts, tokenizer, config.max_len, val_targets
            )
            test_dataset = InsultDataset(
                test_texts, tokenizer, config.max_len, targets=None
            )

            # Dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.train_batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.valid_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.valid_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )

            # Model
            model = InsultModel(model_name, config=config, pretrained=True)
            model.to(device)

            # Optimizer
            optimizer_parameters = get_optimizer_params(
                model, config.learning_rate, config.weight_decay
            )
            optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.learning_rate)

            # Scheduler
            num_train_steps = int(
                len(train_texts) / config.train_batch_size * config.epochs_stage1
            )
            # Adjust epochs based on stage
            epochs = (
                config.epochs_stage1 if stage == "teacher" else config.epochs_stage2
            )

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_train_steps * config.warmup_ratio),
                num_training_steps=num_train_steps,
            )

            # AWP (only for student stage)
            awp = None
            if stage == "student" and config.use_awp:
                awp = AWP(model, optimizer, config)

            # Training Loop
            for epoch in range(epochs):
                if stage == "teacher":
                    avg_loss = train_fn(
                        train_loader, model, optimizer, device, scheduler, epoch, config
                    )
                else:
                    avg_loss = train_fn_awp(
                        train_loader,
                        model,
                        optimizer,
                        device,
                        scheduler,
                        epoch,
                        config,
                        awp,
                    )

            # Inference
            val_loss, val_preds = eval_fn(val_loader, model, device)
            _, test_preds = eval_fn(test_loader, model, device)

            fold_val_preds.append(val_preds)
            fold_test_preds.append(test_preds)

            # Cleanup
            del model, optimizer, scheduler, train_loader, val_loader, test_loader, awp
            torch.cuda.empty_cache()

    return fold_val_preds, fold_test_preds


def main():
    config = Config()
    seed_everything(config.seed)
    logger = get_logger("main")

    logger.info("Starting Corrected Adversarial Knowledge Distillation Ensemble")

    # 1. Load Data
    train_df = load_and_process_data("train", config)
    val_df = load_and_process_data("val", config)
    test_df = load_and_process_data("test", config)

    # ==========================================
    # Stage 1: Teacher Ensemble
    # ==========================================
    logger.info("=== Stage 1: Teacher Ensemble Training ===")
    _, teacher_test_preds_list = run_training(
        config, train_df, val_df, test_df, stage="teacher"
    )

    # Average teacher predictions to create soft targets
    # teacher_test_preds_list is a list of arrays, each shape (N_test,)
    soft_targets = np.mean(teacher_test_preds_list, axis=0)
    logger.info("Soft targets generated.")

    # ==========================================
    # Stage 2: Student Distillation (with AWP)
    # ==========================================
    logger.info("=== Stage 2: Student Distillation Training ===")
    student_val_preds_list, student_test_preds_list = run_training(
        config, train_df, val_df, test_df, stage="student", soft_targets=soft_targets
    )

    # ==========================================
    # Evaluation
    # ==========================================
    # Average student predictions
    final_val_preds = np.mean(student_val_preds_list, axis=0)
    final_test_preds = np.mean(student_test_preds_list, axis=0)

    # Calculate Metric
    val_targets = val_df["Insult"].values
    auc_score = roc_auc_score(val_targets, final_val_preds)

    print(f"Final Validation Metric: {auc_score}")

    # ==========================================
    # Failure Analysis
    # ==========================================
    logger.info("Performing Failure Analysis...")
    val_df["pred"] = final_val_preds
    val_df["error"] = np.abs(val_df["Insult"] - val_df["pred"])
    val_df["char_len"] = val_df["Comment"].apply(len)

    correlation = val_df["error"].corr(val_df["char_len"])
    print(f"Correlation between Error and Comment Length: {correlation:.4f}")

    # ==========================================
    # Submission
    # ==========================================
    threshold = 0.9660591133004925
    if auc_score > threshold:
        logger.info(
            f"Validation score {auc_score} exceeds threshold {threshold}. Generating submission."
        )

        submission = pd.DataFrame()
        # Ensure we use the Date column from test_df if available, or just index
        if "Date" in test_df.columns:
            submission["Date"] = test_df["Date"]

        # The sample submission format usually requires specific columns.
        # Based on sample_submission_null.csv provided in description:
        # It has Insult, Date, Comment. But usually submission just needs ID/Date and Prediction.
        # The prompt says: "Your predictions should be a number in the range [0,1]."
        # And "See 'sample_submissions_null.csv' for the correct format."
        # The sample has 'Insult' column. We will fill that.

        # We'll construct the dataframe to match the sample structure as best as possible
        # Assuming we need to fill the 'Insult' column with probabilities.

        # Load sample submission to get exact format if possible, otherwise reconstruct
        sample_path = "./input/sample_submission_null.csv"
        if os.path.exists(sample_path):
            sub_df = pd.read_csv(sample_path)
            # Ensure alignment. The test.csv and sample_submission usually align by row.
            if len(sub_df) == len(final_test_preds):
                sub_df["Insult"] = final_test_preds
                sub_df.to_csv(config.submission_path, index=False)
                logger.info(f"Submission saved to {config.submission_path}")
            else:
                logger.warning(
                    "Sample submission length mismatch. Creating new dataframe."
                )
                submission = test_df.copy()
                submission["Insult"] = final_test_preds
                submission.to_csv(config.submission_path, index=False)
        else:
            # Fallback
            submission = test_df.copy()
            submission["Insult"] = final_test_preds
            submission.to_csv(config.submission_path, index=False)
            logger.info(f"Submission saved to {config.submission_path}")

    else:
        logger.warning(
            f"Validation score {auc_score} did not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
