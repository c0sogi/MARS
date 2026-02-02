import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import warnings

# Import library modules
from library.config import ModelConfig
from library.utils import set_seed, get_score
from library.dataset import load_dataset_df, InsultDataset
from library.model import InsultModel
from library.engine import fit

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_inference(model, dataloader, device):
    """
    Runs inference on a dataloader using the provided model.
    Returns raw probabilities.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)

            outputs = model(input_ids, attention_mask)
            # Apply sigmoid to get probabilities in [0, 1]
            probs = torch.sigmoid(outputs.view(-1))
            preds.append(probs.cpu().numpy())
    return np.concatenate(preds)


def main():
    # 1. Configuration and Setup
    config = ModelConfig()
    set_seed(config.seed)

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # 2. Load Data
    # Load training data for Cross-Validation
    df_train_full = load_dataset_df(config, split="train", load_cached_data=True)

    # Load hold-out validation data for final evaluation
    df_holdout = load_dataset_df(config, split="val", load_cached_data=True)

    # Load test data for submission
    df_test = load_dataset_df(config, split="test", load_cached_data=True)

    # 3. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 4. Stratified K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )

    # Prepare data for splitting
    X = df_train_full["Comment"].values
    y = df_train_full["Insult"].values

    # Store model paths for ensemble inference
    model_paths = []

    print(f"Starting {config.n_folds}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*20} Fold {fold+1} / {config.n_folds} {'='*20}")

        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create Datasets
        train_dataset = InsultDataset(X_train, tokenizer, config.max_len, y_train)
        val_dataset = InsultDataset(X_val, tokenizer, config.max_len, y_val)

        # Create Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        # Initialize Model
        model = InsultModel(config)
        model.to(config.device)

        # Optimizer and Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Calculate steps for scheduler
        num_train_steps = int(
            len(train_loader) * config.epochs / config.accumulation_steps
        )
        num_warmup_steps = int(num_train_steps * config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Train the model
        model, best_auc = fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            config.device,
            config,
            fold,
        )

        # Store path to best model
        model_path = os.path.join(config.working_dir, f"model_fold_{fold}.bin")
        model_paths.append(model_path)

        # Cleanup to free GPU memory
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()

    # 5. Evaluation on Hold-out Set (Ensemble)
    print(f"\n{'='*20} Final Evaluation on Hold-out Set {'='*20}")

    holdout_dataset = InsultDataset(
        df_holdout["Comment"].values,
        tokenizer,
        config.max_len,
        df_holdout["Insult"].values,
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Accumulate predictions from all fold models
    ensemble_preds = np.zeros(len(df_holdout))

    for i, path in enumerate(model_paths):
        print(f"Inferencing with model fold {i}...")
        model = InsultModel(config)
        model.load_state_dict(torch.load(path, map_location=config.device))
        model.to(config.device)

        preds = run_inference(model, holdout_loader, config.device)
        ensemble_preds += preds

        del model
        torch.cuda.empty_cache()

    # Average predictions
    ensemble_preds /= config.n_folds

    # Calculate Metric
    final_auc = get_score(df_holdout["Insult"].values, ensemble_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print(f"\n{'='*20} Failure Analysis {'='*20}")
    y_true = df_holdout["Insult"].values
    errors = np.abs(y_true - ensemble_preds)

    # Compute features for correlation
    comments = df_holdout["Comment"].values
    char_lens = np.array([len(str(t)) for t in comments])
    word_lens = np.array([len(str(t).split()) for t in comments])

    # Correlations
    corr_char = np.corrcoef(errors, char_lens)[0, 1]
    corr_word = np.corrcoef(errors, word_lens)[0, 1]

    print(f"Correlation between Error and Char Length: {corr_char:.4f}")
    print(f"Correlation between Error and Word Length: {corr_word:.4f}")

    # 7. Submission
    threshold = 0.9639408866995074
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) > threshold ({threshold}). Generating submission..."
        )

        test_dataset = InsultDataset(
            df_test["Comment"].values, tokenizer, config.max_len, labels=None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        test_ensemble_preds = np.zeros(len(df_test))

        for i, path in enumerate(model_paths):
            print(f"Inferencing test set with model fold {i}...")
            model = InsultModel(config)
            model.load_state_dict(torch.load(path, map_location=config.device))
            model.to(config.device)

            preds = run_inference(model, test_loader, config.device)
            test_ensemble_preds += preds

            del model
            torch.cuda.empty_cache()

        test_ensemble_preds /= config.n_folds

        # Create submission dataframe
        submission = pd.DataFrame(
            {
                "Insult": test_ensemble_preds,
                "Date": df_test["Date"],
                "Comment": df_test["Comment"],
            }
        )

        submission_path = os.path.join(config.output_dir, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric ({final_auc}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
