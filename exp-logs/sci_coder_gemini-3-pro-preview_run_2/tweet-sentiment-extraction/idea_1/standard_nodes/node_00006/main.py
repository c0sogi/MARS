import torch
import pandas as pd
import numpy as np
import os
import sys

# Import from the provided library
import importlib
import library.config as config

importlib.reload(config)  # Cite debug_lesson_1

import library.data_loader

importlib.reload(library.data_loader)  # Cite debug_lesson_1
from library.data_loader import get_loaders

import library.model

importlib.reload(library.model)  # Cite debug_lesson_1
from library.model import TransformerPointerNetwork

import library.engine

importlib.reload(library.engine)  # Cite debug_lesson_1
from library.engine import train_fn, eval_fn, predict_fn, decode_span

from library.utils import jaccard


def run_pipeline():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Override config defaults for a faster run while utilizing A100 memory
    config.NUM_EPOCHS = 5
    config.BATCH_SIZE = 256  # Increase batch size for A100

    print(
        f"Configuration: Epochs={config.NUM_EPOCHS}, Batch Size={config.BATCH_SIZE}, Device={config.DEVICE}"
    )

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = get_loaders(load_cached_data=True)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = BiGRUPointerNetwork(len(vocab)).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_jaccard = 0.0

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, config.DEVICE)

        # Validate
        val_loss, val_jaccard = eval_fn(val_loader, model, config.DEVICE)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
        )

        # Save Best Model
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            # print(f"  -> Model saved (New Best Jaccard: {best_jaccard:.4f})")

    # ==========================================
    # 5. Final Validation & Metric
    # ==========================================
    print("\n--- Final Evaluation ---")
    # Load best model for final evaluation
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

    # Compute final metric on validation set
    _, final_val_jaccard = eval_fn(val_loader, model, config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_jaccard}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")
    model.eval()

    errors = []
    lengths = []
    sentiments = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            start_positions = batch["start_idx"].to(config.DEVICE)
            end_positions = batch["end_idx"].to(config.DEVICE)
            texts = batch["text"]
            offsets = batch["offset_mapping"].numpy()

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask)
            start_preds, end_preds = decode_span(
                start_logits, end_logits, attention_mask
            )

            # Analyze per sample
            for k in range(len(texts)):
                text = texts[k]

                # Reconstruct Prediction
                pred_str = get_selected_text(
                    text, start_preds[k], end_preds[k], offsets[k]
                )

                # Reconstruct Ground Truth
                gt_str = get_selected_text(
                    text, start_positions[k].item(), end_positions[k].item(), offsets[k]
                )

                # Calculate Jaccard
                score = jaccard(pred_str, gt_str)

                # Store data for correlation
                errors.append(1.0 - score)
                lengths.append(len(text.split()))  # Use word count for length analysis

                # Map sentiment string to code for analysis consistency
                sent_str = batch["sentiment"][k]
                sent_code = config.SENTIMENT_MAP[sent_str]
                sentiments.append(sent_code)

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {"error": errors, "input_length": lengths, "sentiment_code": sentiments}
    )

    # Calculate correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Model Error (1-Jaccard) and Input Features:")
    print(correlations)

    # Identify highest error segment
    mean_error_by_sentiment = df_analysis.groupby("sentiment_code")["error"].mean()
    print("\nMean Error by Sentiment Code (0:neg, 1:neu, 2:pos):")
    print(mean_error_by_sentiment)

    # ==========================================
    # 7. Submission
    # ==========================================
    print("\n--- Generating Submission ---")
    predict_fn(test_loader, model, config.DEVICE)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
