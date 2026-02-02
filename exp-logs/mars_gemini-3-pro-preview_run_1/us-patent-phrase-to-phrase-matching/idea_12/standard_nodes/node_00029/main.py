import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, compute_score, get_llrd_optimizer_params
from library.data import (
    get_tokenizer_and_resize,
    make_dataloaders,
    make_test_dataloader,
    PearsonDataset,
)
from library.model import CustomDeberta
from library.engine import train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # 1. Configuration and Setup
    # We use 2 epochs to ensure the pipeline completes within the time limit
    # while providing a strong baseline.
    config = Config(epochs=2)

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)

    print(
        f"Configuration: Epochs={config.epochs}, Folds={config.n_folds}, Device={config.device}"
    )

    # 2. Prepare Tokenizer (Atomic Contexts)
    print("Initializing Tokenizer and resizing for Atomic Contexts...")
    tokenizer = get_tokenizer_and_resize(config)

    # 3. Stratified Group K-Fold Training
    print("Starting Stratified Group K-Fold Training...")

    # Store paths to saved checkpoints for ensemble inference
    model_paths = []

    for fold in range(config.n_folds):
        print(f"\n[Fold {fold}/{config.n_folds - 1}]")

        # Create DataLoaders for the current fold
        train_loader, val_loader = make_dataloaders(config, tokenizer, fold=fold)

        # Initialize Model
        model = CustomDeberta(config, tokenizer)
        model.to(config.device)

        # Optimizer with Layer-wise Learning Rate Decay (LLRD)
        optimizer_params = get_llrd_optimizer_params(
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            llrd_decay=config.llrd_decay,
        )
        optimizer = AdamW(optimizer_params, lr=config.learning_rate, eps=config.eps)

        # Learning Rate Scheduler
        num_training_steps = len(train_loader) * config.epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training Loop
        best_fold_score = -1.0
        best_model_path = os.path.join(config.output_dir, f"model_fold_{fold}.pth")

        for epoch in range(config.epochs):
            # Train one epoch
            train_loss = train_one_epoch(
                model, optimizer, scheduler, train_loader, config.device, epoch, config
            )

            # Validate on fold validation set
            val_loss, val_score = validate(model, val_loader, config.device, config)

            # Save Checkpoint if improved
            if val_score > best_fold_score:
                best_fold_score = val_score
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} Best Score: {best_fold_score:.4f}")
        model_paths.append(best_model_path)

        # Cleanup to save memory for next fold
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Hold-out Validation Evaluation
    print("\n=== Hold-out Validation Evaluation ===")

    # Load Hold-out Validation Set
    df_val = pd.read_csv(config.val_path)
    val_dataset = PearsonDataset(df_val, tokenizer, config.max_length, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Helper function for Ensemble Inference
    def run_inference(loader, paths):
        all_preds = []
        # Iterate over each fold model
        for path in paths:
            model = CustomDeberta(config, tokenizer)
            model.load_state_dict(torch.load(path, map_location=config.device))
            model.to(config.device)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(config.device)
                    attention_mask = batch["attention_mask"].to(config.device)
                    token_type_ids = batch.get("token_type_ids", None)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(config.device)

                    # Mixed precision inference
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids, attention_mask, token_type_ids)

                    fold_preds.append(outputs.logits.view(-1).cpu().float().numpy())

            all_preds.append(np.concatenate(fold_preds))

            del model
            torch.cuda.empty_cache()

        # Average predictions (Ensemble)
        avg_preds = np.mean(all_preds, axis=0)
        return avg_preds

    # Generate Validation Predictions
    val_preds = run_inference(val_loader, model_paths)
    val_targets = df_val["score"].values

    # Compute Final Metric
    final_metric = compute_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_val["prediction"] = val_preds
    df_val["abs_error"] = (df_val["score"] - df_val["prediction"]).abs()

    # Feature Engineering for Analysis
    df_val["len_anchor"] = df_val["anchor"].astype(str).apply(len)
    df_val["len_target"] = df_val["target"].astype(str).apply(len)
    df_val["len_diff"] = (df_val["len_anchor"] - df_val["len_target"]).abs()

    # Calculate Correlations
    analysis_cols = ["abs_error", "score", "len_anchor", "len_target", "len_diff"]
    corr_matrix = df_val[analysis_cols].corr()
    error_correlations = corr_matrix["abs_error"].drop("abs_error")

    print("Correlation between Error Magnitude and Features:")
    print(error_correlations)

    # 6. Test Submission
    TARGET_THRESHOLD = 0.8673

    if final_metric > TARGET_THRESHOLD:
        print(f"\nMetric {final_metric} > {TARGET_THRESHOLD}. Generating Submission...")

        # Create Test Loader
        test_loader = make_test_dataloader(config, tokenizer)

        # Generate Test Predictions
        test_preds = run_inference(test_loader, model_paths)

        # Load Test IDs for submission mapping
        df_test = pd.read_csv(config.test_path)

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": df_test["id"], "score": test_preds})

        # Save
        submission_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(f"\nMetric {final_metric} <= {TARGET_THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
