import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from provided libraries
from library.utils import seed_everything, compute_score
from library.dataset import get_datasets
from library.model import DebertaV3FeatureFused
from library.engine import train_one_epoch, predict


# --- Configuration ---
class Config:
    seed = 42
    model_name = "microsoft/deberta-v3-large"
    output_dir = "./working/models"
    submission_dir = "./submission"
    max_length = 128
    batch_size = 16  # Optimized for A100
    gradient_accumulation_steps = 2  # Effective batch size 32
    max_grad_norm = 1.0
    epochs = 3
    folds = 4  # Increased from 3 for better ensemble
    lr = 1.5e-5  # Slightly lower for stability
    weight_decay = 0.01
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = 0.8633016507628494


def main():
    # 1. Setup
    seed_everything(Config.seed)
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    print(f"Using device: {Config.device}")

    # 2. Data Loading
    # Load metadata and features using the library function
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_ds_full, val_ds_holdout, test_ds = get_datasets(
        tokenizer=tokenizer,
        max_length=Config.max_length,
        load_cached_data=True,
        debug=False,
    )

    # 3. Stratified K-Fold Training
    # We split the main training set into K folds for robust training
    skf = StratifiedKFold(n_splits=Config.folds, shuffle=True, random_state=Config.seed)

    # Labels for stratification (extracted from the dataset)
    y_stratify = train_ds_full.labels

    model_paths = []

    print(f"Starting training: {Config.folds} folds, {Config.epochs} epochs each.")

    for fold, (train_idx, _) in enumerate(
        skf.split(np.zeros(len(y_stratify)), y_stratify)
    ):
        print(f"\n=== Fold {fold} ===")

        # Create Subset for the current fold
        train_sub = Subset(train_ds_full, train_idx)

        train_loader = DataLoader(
            train_sub,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Initialize Model
        model = DebertaV3FeatureFused(
            model_name=Config.model_name,
            num_classes=5,
            num_structural_features=3,
            pretrained=True,
        )
        model.to(Config.device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
        )

        # Adjust total steps for gradient accumulation
        num_update_steps_per_epoch = (
            len(train_loader) // Config.gradient_accumulation_steps
        )
        total_steps = num_update_steps_per_epoch * Config.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        # Training Loop
        for epoch in range(Config.epochs):
            train_one_epoch(
                model,
                optimizer,
                scheduler,
                train_loader,
                Config.device,
                epoch,
                gradient_accumulation_steps=Config.gradient_accumulation_steps,
                max_grad_norm=Config.max_grad_norm,
            )

        # Save Model
        model_path = os.path.join(Config.output_dir, f"model_fold_{fold}.bin")
        torch.save(model.state_dict(), model_path)
        model_paths.append(model_path)

        # Cleanup
        del model, optimizer, scheduler, train_loader
        torch.cuda.empty_cache()

    # 4. Final Validation on Hold-out Set
    print("\nRunning Final Validation on Hold-out Set...")
    val_loader = DataLoader(
        val_ds_holdout,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Ensemble Inference
    ensemble_preds = np.zeros(len(val_ds_holdout))

    for model_path in model_paths:
        model = DebertaV3FeatureFused(
            model_name=Config.model_name,
            num_classes=5,
            num_structural_features=3,
            pretrained=False,
        )
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)

        preds = predict(model, val_loader, Config.device)
        ensemble_preds += preds

        del model
        torch.cuda.empty_cache()

    # Average predictions
    ensemble_preds /= Config.folds

    # Compute Metric
    true_scores = val_ds_holdout.scores
    final_metric = compute_score(true_scores, ensemble_preds)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    errors = np.abs(true_scores - ensemble_preds)

    # Access features from the dataset
    val_df = val_ds_holdout.df
    structural_feats = val_ds_holdout.structural_features

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "norm_levenshtein": structural_feats[:, 0],
            "jaccard_sim": structural_feats[:, 1],
            "len_ratio": structural_feats[:, 2],
            "anchor_len": val_df["anchor"].fillna("").astype(str).str.len(),
            "target_len": val_df["target"].fillna("").astype(str).str.len(),
        }
    )

    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    if final_metric > Config.threshold:
        print("\nMetric passed threshold. Generating submission...")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        test_ensemble_preds = np.zeros(len(test_ds))

        for model_path in model_paths:
            model = DebertaV3FeatureFused(
                model_name=Config.model_name,
                num_classes=5,
                num_structural_features=3,
                pretrained=False,
            )
            model.load_state_dict(torch.load(model_path, map_location=Config.device))
            model.to(Config.device)

            preds = predict(model, test_loader, Config.device)
            test_ensemble_preds += preds

            del model
            torch.cuda.empty_cache()

        test_ensemble_preds /= Config.folds

        submission_df = pd.DataFrame(
            {"id": test_ds.df["id"], "score": test_ensemble_preds}
        )

        sub_path = os.path.join(Config.submission_dir, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nMetric {final_metric} did not pass threshold {Config.threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
