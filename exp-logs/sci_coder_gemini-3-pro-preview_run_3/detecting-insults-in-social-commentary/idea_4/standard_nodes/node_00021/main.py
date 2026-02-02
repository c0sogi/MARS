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

    # 4. Single Model Training (Cite solution_lesson_node_00018: Maximize training data)
    print(f"Starting Single Model Training...")

    # Use full training set
    X_train = df_train_full["Comment"].values
    y_train = df_train_full["Insult"].values

    # Use hold-out set for validation
    X_val = df_holdout["Comment"].values
    y_val = df_holdout["Insult"].values

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
    num_train_steps = int(len(train_loader) * config.epochs / config.accumulation_steps)
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
        fold=0,
    )

    # Store path to best model
    model_path = os.path.join(config.working_dir, "model_fold_0.bin")

    # 5. Evaluation on Hold-out Set
    # Since we validated on the hold-out set during training, best_auc is the result.
    # However, to be consistent with the pipeline, we can re-inference or just use best_auc.
    # We'll re-inference to ensure we have the predictions vector for failure analysis.

    print(f"\n{'='*20} Final Evaluation on Hold-out Set {'='*20}")

    # Reload best model
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    ensemble_preds = run_inference(model, val_loader, config.device)
    final_auc = get_score(y_val, ensemble_preds)

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
    threshold = 0.9639490968801314
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

        print(f"Inferencing test set...")
        # Model is already loaded and on device
        test_ensemble_preds = run_inference(model, test_loader, config.device)

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
