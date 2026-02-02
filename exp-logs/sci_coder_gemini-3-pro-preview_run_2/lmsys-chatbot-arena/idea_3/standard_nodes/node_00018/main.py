import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import library modules
from library.config import Config, seed_everything
from library.data_processor import load_data
from library.dataset import ChatbotDataset
from library.model import SiameseDeberta
from library.trainer import Trainer
from library.utils import compute_score, load_checkpoint
from library.inference import run_inference


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Data Loading
    # load_cached_data=True allows using pre-processed parquet files if they exist
    df_train, df_val, df_test = load_data(load_cached_data=True)

    # 3. Prepare Datasets & DataLoaders
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    print("Creating Datasets...")
    train_dataset = ChatbotDataset(df_train, tokenizer, is_test=False)
    val_dataset = ChatbotDataset(df_val, tokenizer, is_test=False)

    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 4. Model Initialization
    print("Initializing Model...")
    # meta_dim=3 for [prompt_len, response_a_len, response_b_len]
    model = SiameseDeberta(model_name=Config.MODEL_NAME, num_classes=3, meta_dim=3)

    # 5. Training
    trainer = Trainer(model, device=Config.DEVICE)
    print("Starting Training...")
    trainer.train(train_loader, val_loader, epochs=Config.EPOCHS)

    # 6. Validation Assessment
    print("\n--- Validation Assessment ---")
    # Load the best checkpoint saved during training
    load_checkpoint(model, path=Config.MODEL_SAVE_PATH, device=Config.DEVICE)

    # Generate predictions on validation set
    val_probs = trainer.predict(val_loader)

    # Get ground truth targets (numpy array)
    val_targets = df_val[["winner_model_a", "winner_model_b", "winner_tie"]].values

    # Compute Score
    final_score = compute_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_score}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample Cross Entropy Loss
    # Clip predictions to avoid log(0)
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    per_sample_loss = -np.sum(val_targets * np.log(val_probs_clipped), axis=1)

    # Calculate correlation with meta-features
    # We use the scaled meta-features present in df_val
    meta_cols = ["meta_prompt_len", "meta_a_len", "meta_b_len"]
    print("Correlation between Error Magnitude and Input Features:")

    for col in meta_cols:
        if col in df_val.columns:
            # Compute Pearson correlation
            corr = np.corrcoef(df_val[col].values, per_sample_loss)[0, 1]
            print(f"  {col}: {corr}")
        else:
            print(f"  {col}: Not found in dataframe")

    # 8. Submission Generation
    THRESHOLD = 1.0102717496437368
    if final_score < THRESHOLD:
        print(
            f"\nValidation score ({final_score}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission for test set...")

        # Run inference pipeline
        # This re-loads data and model, but ensures clean execution environment
        run_inference(load_cached_data=True, device=Config.DEVICE)

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score ({final_score}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
