import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.dataset import load_processed_train_data, ToxicityDataset
from library.model import MultiTaskRoBERTa
from library.trainer import Trainer
from library.metrics import compute_final_score


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(val_df, preds):
    """
    Analyzes model errors on the validation set.
    Correlates error magnitude with text length and identity attributes.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error Magnitude
    # Target >= 0.5 is positive class
    y_true = (val_df[Config.TARGET_COL].values >= 0.5).astype(int)
    errors = np.abs(y_true - preds)

    analysis_df = val_df.copy()
    analysis_df["error"] = errors
    analysis_df["text_len"] = analysis_df[Config.TEXT_COL].fillna("").str.len()

    # 1. Correlation with Text Length
    len_corr = analysis_df["error"].corr(analysis_df["text_len"])
    print(f"Correlation (Error vs Text Length): {len_corr}")

    # 2. Correlation with Identity Attributes
    print("Correlation (Error vs Identity Presence):")
    for col in Config.IDENTITY_COLUMNS:
        if col in analysis_df.columns:
            # Fill NaNs with 0 for correlation calculation
            id_series = analysis_df[col].fillna(0)
            # Only calculate if we have some non-zero values
            if id_series.sum() > 0:
                id_corr = analysis_df["error"].corr(id_series)
                print(f"  {col}: {id_corr}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading & Preparation
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Train Data
    # Using load_cached_data=True as requested
    print("Loading training data...")
    train_df = load_processed_train_data(load_cached_data=True, debug=False)

    # Fast Baseline: Subsample training data
    # We use 50,000 samples to ensure execution within 2 hours
    SAMPLE_SIZE = 50000
    if len(train_df) > SAMPLE_SIZE:
        print(f"Subsampling training data to {SAMPLE_SIZE} rows for fast baseline...")
        train_df = train_df.sample(n=SAMPLE_SIZE, random_state=Config.SEED).reset_index(
            drop=True
        )

    # Load Validation Data (Full set required for accurate metric)
    print("Loading validation data...")
    val_df = pd.read_csv(Config.VAL_PATH)

    # Create Datasets
    train_dataset = ToxicityDataset(train_df, tokenizer, mode="train")
    val_dataset = ToxicityDataset(val_df, tokenizer, mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = MultiTaskRoBERTa(pretrained=True)
    model.to(device)

    # 4. Optimizer & Scheduler
    # Fast Baseline: Train for 2 epochs
    EPOCHS = 2

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # 5. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        val_df=val_df,
        device=device,
    )

    trainer.fit(epochs=EPOCHS)

    # 6. Final Evaluation
    print("\nRunning final evaluation on validation set...")
    # Load best model saved by Trainer
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Generate predictions on validation set
    _, val_preds = trainer.predict(val_loader)

    # Compute Final Metric
    final_score, metrics_summary = compute_final_score(val_df, val_preds)

    # REQUIRED OUTPUT: Print full precision metric
    print(f"Final Validation Metric: {final_score}")

    # 7. Failure Analysis
    perform_failure_analysis(val_df, val_preds)

    # 8. Submission Generation
    THRESHOLD = 0.9378314975158606

    if final_score > THRESHOLD:
        print(
            f"\nValidation score ({final_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_df = pd.read_csv(Config.TEST_PATH)
        test_dataset = ToxicityDataset(test_df, tokenizer, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Generate Predictions
        ids, predictions = trainer.predict(test_loader)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": ids, "prediction": predictions})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score ({final_score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
