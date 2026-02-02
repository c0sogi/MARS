import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, logging

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_pearson
from library.data import prepare_loaders
from library.model import CustomModel
from library.engine import fit
from library.inference import predict


def main():
    # 1. Configuration and Setup
    # Suppress transformers warnings for cleaner output
    logging.set_verbosity_error()

    # Override Config for Fast Baseline
    # Reducing epochs to 2 ensures completion within 2 hours while allowing convergence
    # on the powerful A100 GPU.
    Config.epochs = 2

    # Set seeds for reproducibility
    set_seed(Config.seed)

    print(f"Running on device: {Config.device}")

    # 2. Data Preparation
    print("Initializing Tokenizer and DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load data (using cache if available)
    # We use debug=False to train on the full dataset to meet the score threshold.
    # The A100 is capable of processing the full dataset quickly.
    train_loader, val_loader, _ = prepare_loaders(
        tokenizer=tokenizer, load_cached_data=True, debug=False
    )

    # 3. Training
    print("Starting Training...")
    fit(train_loader, val_loader)

    # 4. Final Validation Assessment & Failure Analysis
    print("\nStarting Final Validation Assessment...")

    # Load the best model saved during training
    if not os.path.exists(Config.model_path):
        raise FileNotFoundError("Model file not found. Training may have failed.")

    model = CustomModel()
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()

    # Generate predictions on the validation set
    all_preds = []
    all_labels = []

    # Re-iterate validation loader to get element-wise predictions
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.device)
            mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            outputs = model(input_ids, mask).view(-1)

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # Clip predictions to valid range
    preds = np.clip(preds, 0, 1)

    # Compute and print metric
    final_metric = compute_pearson(preds, labels)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to access features
    df_val = pd.read_csv(Config.val_metadata_path)

    # Ensure alignment (dataloaders without drop_last should match metadata length)
    if len(df_val) != len(preds):
        print(
            f"Note: Adjusting metadata length ({len(df_val)}) to match predictions ({len(preds)})"
        )
        df_val = df_val.iloc[: len(preds)].copy()

    # Calculate Error
    df_val["pred"] = preds
    df_val["label"] = labels
    df_val["error"] = (df_val["pred"] - df_val["label"]).abs()

    # Extract Features for Correlation Analysis
    # 1. Lengths
    df_val["anchor_len"] = df_val["anchor"].astype(str).apply(len)
    df_val["target_len"] = df_val["target"].astype(str).apply(len)
    df_val["len_diff"] = (df_val["anchor_len"] - df_val["target_len"]).abs()

    # 2. Jaccard Similarity
    def calc_jaccard(row):
        a = set(str(row["anchor"]).lower().split())
        b = set(str(row["target"]).lower().split())
        u = len(a.union(b))
        return len(a.intersection(b)) / u if u > 0 else 0.0

    df_val["jaccard"] = df_val.apply(calc_jaccard, axis=1)

    # Compute Correlations
    analysis_features = ["anchor_len", "target_len", "len_diff", "jaccard"]
    correlations = df_val[analysis_features].corrwith(df_val["error"])

    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # 5. Submission
    # Threshold defined in task
    THRESHOLD = 0.8288092510484422

    if final_metric > THRESHOLD:
        print(f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}).")
        print("Generating submission for test set...")
        predict(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
