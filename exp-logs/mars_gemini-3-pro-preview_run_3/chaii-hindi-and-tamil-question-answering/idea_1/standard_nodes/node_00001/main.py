import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, logging

# Import provided library components
from library.config import Config
from library.utils import jaccard
from library.model import QATokenClassifier
from library.data_loader import prepare_data
from library.trainer import Trainer

# Suppress excessive transformer warnings
logging.set_verbosity_error()


def main():
    # 1. Configuration and Setup
    # Limit epochs for a fast baseline execution
    Config.EPOCHS = 5
    Config.setup()

    print("Initializing Fast Baseline Run...")

    # 2. Data Loading
    # Load processed datasets (using cache if available)
    train_dataset, val_dataset, test_dataset = prepare_data(load_cached_data=True)

    # Load raw metadata for alignment and analysis
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = QATokenClassifier(Config.MODEL_NAME)

    # Move model to device
    device = Config.DEVICE
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Initialize Trainer
    trainer = Trainer(model, tokenizer, device)

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")
    best_val_score = -1.0

    for epoch in range(Config.EPOCHS):
        train_loss = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)

        # Validate using the trainer's method which returns average Jaccard
        val_score = trainer.validate(val_loader, df_val)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Jaccard: {val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            trainer.save_model(Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved (Score: {best_val_score:.4f})")

    # 5. Final Validation & Metric Reporting
    print("\nLoading best model for final evaluation...")
    trainer.load_model(Config.MODEL_SAVE_PATH)

    # We use predict() on validation set to get individual strings for analysis
    # Note: We pass df_val to align IDs, though predict returns IDs and Preds
    val_ids, val_preds = trainer.predict(val_loader, df_val)

    # Align predictions with Ground Truth
    # Create a map from ID to prediction
    pred_map = dict(zip(val_ids, val_preds))

    # Calculate Jaccard for every sample in the validation dataframe
    scores = []
    for idx, row in df_val.iterrows():
        uid = row["id"]
        gt = row["answer_text"]
        pred = pred_map.get(uid, "")
        score = jaccard(gt, pred)
        scores.append(score)

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = df_val.copy()
    df_analysis["jaccard"] = scores
    df_analysis["error"] = 1.0 - df_analysis["jaccard"]

    # Feature Engineering for Analysis
    df_analysis["context_len"] = df_analysis["context"].astype(str).apply(len)
    df_analysis["question_len"] = df_analysis["question"].astype(str).apply(len)
    df_analysis["is_tamil"] = (df_analysis["language"] == "tamil").astype(int)

    # Calculate Correlations
    correlations = df_analysis[
        ["error", "context_len", "question_len", "is_tamil"]
    ].corr()["error"]

    print("Correlation between Error (1-Jaccard) and Input Features:")
    print(f"  Context Length:  {correlations['context_len']:.4f}")
    print(f"  Question Length: {correlations['question_len']:.4f}")
    print(f"  Language (Tamil): {correlations['is_tamil']:.4f}")

    # 7. Submission Generation
    print("\nGenerating submission for test set...")
    test_ids, test_preds = trainer.predict(test_loader, df_test)

    submission_df = pd.DataFrame({"id": test_ids, "PredictionString": test_preds})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
