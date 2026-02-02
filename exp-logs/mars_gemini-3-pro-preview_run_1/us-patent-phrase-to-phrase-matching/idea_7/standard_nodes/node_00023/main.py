import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.data import prepare_data, PhraseDataset
from library.trainer import run_fold
from library.inference import run_inference, inference_fn
from library.model import DebertaV3Regressor
from library.utils import seed_everything, compute_score


def main():
    # 1. Setup and Configuration
    seed_everything(Config.seed)

    # Override Config for Fast Baseline execution
    # Reducing epochs to 2 ensures the script completes quickly while allowing the model to converge.
    Config.epochs = 2

    print("=" * 40)
    print(" STARTING FAST BASELINE PIPELINE")
    print("=" * 40)

    # 2. Data Preparation
    # prepare_data handles context injection (mapping CPC codes to text) and caching.
    # It returns a merged dataframe of all labeled data.
    print("\n[Data Preparation]")
    df_all, df_test = prepare_data(load_cached_data=True)

    # Reconstruct the fixed Hold-out Validation Split using metadata files.
    # This ensures we strictly use the designated validation set for metric calculation.
    meta_train = pd.read_csv(Config.train_path)
    meta_val = pd.read_csv(Config.val_path)

    train_ids = set(meta_train["id"])
    val_ids = set(meta_val["id"])

    # Filter the processed dataframe to create our specific splits
    train_data = df_all[df_all["id"].isin(train_ids)].reset_index(drop=True)
    valid_data = df_all[df_all["id"].isin(val_ids)].reset_index(drop=True)

    print(f"Train Set Size: {len(train_data)}")
    print(f"Valid Set Size: {len(valid_data)}")

    # 3. Training
    # We train a single model (Fold 0) on the fixed training set.
    print("\n[Training]")
    # run_fold trains the model and saves it to ./working/idea_7/models/model_fold_0.pth
    run_fold(0, train_data, valid_data)

    # 4. Validation & Failure Analysis
    print("\n[Validation & Failure Analysis]")

    # Load the trained model for evaluation
    device = Config.device
    model_path = os.path.join(Config.models_dir, "model_fold_0.pth")

    model = DebertaV3Regressor(Config.model_name, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Prepare DataLoader for the validation set
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    valid_dataset = PhraseDataset(
        valid_data, tokenizer, max_length=Config.max_length, is_train=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Generate predictions
    val_preds = inference_fn(model, valid_loader, device)
    val_labels = valid_data["score"].values

    # Compute and print the required metric
    final_metric = compute_score(val_labels, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between error magnitude and input features
    valid_data["pred"] = val_preds
    valid_data["error"] = (valid_data["score"] - valid_data["pred"]).abs()

    # Generate meta-features for analysis
    valid_data["len_anchor"] = valid_data["anchor"].astype(str).apply(len)
    valid_data["len_target"] = valid_data["target"].astype(str).apply(len)
    valid_data["len_context"] = (
        valid_data["context_text"].fillna("").astype(str).apply(len)
    )

    # Compute correlations
    features = ["len_anchor", "len_target", "len_context"]
    correlations = valid_data[["error"] + features].corr()["error"].drop("error")

    print("\nCorrelation between Error and Input Features:")
    print(correlations)

    # 5. Submission
    print("\n[Submission Generation]")
    if final_metric > 0.8673:
        print(f"Metric {final_metric:.5f} > 0.8673. Generating submission...")

        # We temporarily set n_folds to 1 so run_inference only looks for model_fold_0.pth
        Config.n_folds = 1

        # Execute inference on the test set
        run_inference()
    else:
        print(
            f"Metric {final_metric:.5f} did not meet the threshold (0.8673). Skipping submission."
        )


if __name__ == "__main__":
    main()
