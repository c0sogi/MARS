import torch
import pandas as pd
import numpy as np
import os
import sys

# Import from the provided library
import library.config as config
from library.data_loader import get_loaders
from library.model import TweetModel
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
        val_loss, val_jaccard = eval_fn(val_loader, model, config.DEVICE, tokenizer)

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
    _, final_val_jaccard = eval_fn(val_loader, model, config.DEVICE, tokenizer)

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
            sentiment_ids = batch["sentiment_id"].to(config.DEVICE)
            start_positions = batch["start_idx"].to(config.DEVICE)
            end_positions = batch["end_idx"].to(config.DEVICE)
            texts = batch["text"]

            # Forward pass
            start_logits, end_logits = model(input_ids, sentiment_ids, attention_mask)
            start_preds, end_preds = decode_span(
                start_logits, end_logits, attention_mask
            )

            # Analyze per sample
            for k in range(len(texts)):
                ids = input_ids[k].cpu().numpy()

                # Reconstruct Prediction
                i_p, j_p = start_preds[k], end_preds[k]
                pred_str = tokenizer.decode(
                    ids[i_p : j_p + 1], skip_special_tokens=True
                )

                # Reconstruct Ground Truth
                i_t, j_t = start_positions[k].item(), end_positions[k].item()
                gt_str = tokenizer.decode(ids[i_t : j_t + 1], skip_special_tokens=True)

                # Calculate Jaccard
                score = jaccard(pred_str, gt_str)

                # Store data for correlation
                errors.append(1.0 - score)  # Error is inverse of Jaccard
                lengths.append(len(tokens))
                sentiments.append(sentiment_ids[k].item())

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
    predict_fn(test_loader, model, config.DEVICE, tokenizer)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
