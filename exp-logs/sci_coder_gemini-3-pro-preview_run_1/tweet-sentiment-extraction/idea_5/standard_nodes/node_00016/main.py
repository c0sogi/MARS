import sys
import os
import pandas as pd
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup, AdamW

# Import from provided library
from library.config import Config, set_seed
from library.data import get_loaders
from library.model import SentimentModel
from library.engine import get_optimizer_params, train_fn, eval_fn, predict_fn
from library.utils import jaccard


def main():
    # 1. Setup and Config Overrides
    set_seed(Config.seed)

    # Override Config for Fast Baseline while ensuring performance
    # 5 Epochs is sufficient for DeBERTa to converge on this size of data
    Config.epochs = 5
    Config.train_batch_size = 32  # Increase batch size for A100 efficiency

    print(f"Device: {Config.device}")
    print(f"Epochs: {Config.epochs}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(
        debug=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = SentimentModel()
    model.to(Config.device)

    # 4. Optimizer and Scheduler
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate * 5,  # Higher LR for custom heads
        weight_decay=Config.weight_decay,
    )

    optimizer = AdamW(optimizer_parameters, lr=Config.learning_rate, eps=1e-6)

    num_train_steps = int(len(train_loader) * Config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    best_jaccard = 0.0

    print("Starting training...")
    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(
            train_loader, model, optimizer, epoch, scheduler, Config.device
        )

        # Validate
        val_jaccard = eval_fn(val_loader, model, Config.device)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.5f} | Val Jaccard: {val_jaccard:.5f}"
        )

        # Save Best
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), Config.model_save_path)

    # 6. Final Evaluation
    print("\n--- Final Evaluation ---")
    # Load best model
    model.load_state_dict(
        torch.load(Config.model_save_path, map_location=Config.device)
    )
    model.to(Config.device)
    model.eval()

    final_metric = eval_fn(val_loader, model, Config.device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Get predictions for validation set
    val_preds = predict_fn(val_loader, model, Config.device)

    # Load validation metadata to get ground truth and features
    val_df = pd.read_csv(Config.val_path)

    # Ensure alignment (loaders preserve order)
    # Calculate Jaccard per sample
    val_df["prediction"] = val_preds
    val_df["score"] = val_df.apply(
        lambda x: jaccard(x["prediction"], x["selected_text"]), axis=1
    )
    val_df["error"] = 1.0 - val_df["score"]
    val_df["text_len"] = val_df["text"].astype(str).apply(len)

    # Correlation with Text Length
    corr_len = val_df["error"].corr(val_df["text_len"])
    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")

    # Error by Sentiment
    print("Average Error by Sentiment:")
    print(val_df.groupby("sentiment")["error"].mean())

    # 8. Submission
    THRESHOLD = 0.7043342108129372

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = predict_fn(test_loader, model, Config.device)

        # Load test metadata for IDs
        test_df = pd.read_csv(Config.test_path)

        submission = pd.DataFrame(
            {"textID": test_df["textID"], "selected_text": test_preds}
        )

        # Ensure quoting is handled by pandas to_csv
        submission.to_csv(Config.output_submission_path, index=False)
        print(f"Submission saved to {Config.output_submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
