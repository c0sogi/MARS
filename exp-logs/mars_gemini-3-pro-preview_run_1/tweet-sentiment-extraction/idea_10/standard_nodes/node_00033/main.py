import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm

# Import from provided library
from library.config import Config
from library.utils import seed_everything, normalize_text, jaccard
from library.data import get_loaders
from library.model import SentimentConditionedDeberta
from library.engine import train_fn, eval_fn


def get_best_span(start_logits, end_logits, text, offsets):
    """
    Decodes the best span by maximizing start_logit + end_logit
    subject to start_index <= end_index.
    """
    # Convert to tensor for vectorized operations if numpy
    if isinstance(start_logits, np.ndarray):
        start_logits = torch.tensor(start_logits)
    if isinstance(end_logits, np.ndarray):
        end_logits = torch.tensor(end_logits)

    seq_len = len(start_logits)

    # Create sum matrix: [seq_len, seq_len] where (i, j) = start[i] + end[j]
    sum_matrix = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)

    # Mask invalid spans (where end_index < start_index)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0)
    sum_matrix = sum_matrix * mask + (1 - mask) * -1e9

    # Find the indices of the maximum score
    best_idx = torch.argmax(sum_matrix).item()
    start_idx = best_idx // seq_len
    end_idx = best_idx % seq_len

    # Map token indices to character offsets
    # Offsets are typically (start_char, end_char)
    # If the model predicts special tokens (0,0), we might get empty strings.
    # We handle this by checking bounds.
    if start_idx >= len(offsets) or end_idx >= len(offsets):
        return text

    start_char = offsets[start_idx][0]
    end_char = offsets[end_idx][1]

    # If offsets are (0,0) (e.g. [CLS]), it might result in empty string if both are 0.
    # However, usually the model learns to point to valid text tokens.
    # If the extracted span is empty or invalid, fallback to full text is a safe heuristic,
    # though technically for pos/neg we expect a span.

    pred_text = text[start_char:end_char]

    # Fallback if prediction is empty
    if len(pred_text.strip()) == 0:
        return text

    return pred_text


def run_inference(model, loader, df, device):
    """
    Runs inference on a dataset (val or test).
    Applies the neutral strategy and model decoding.
    """
    # Get raw logits
    _, start_logits_all, end_logits_all = eval_fn(loader, model, device)

    predictions = []

    # Iterate through the dataframe and corresponding logits
    # Note: loader and df must be aligned. get_loaders(shuffle=False) ensures this.

    # We need the offsets from the dataset.
    # Since eval_fn only returns logits, we need to access offsets from the loader's dataset.
    # The dataset stores offsets in memory.
    dataset_offsets = loader.dataset.offsets

    print(f"Generating predictions for {len(df)} samples...")

    for idx, row in df.iterrows():
        text = str(row["text"])
        # IMPORTANT: Normalize text to match the tokenizer offsets
        normalized_text = normalize_text(text)
        sentiment = row["sentiment"]

        if sentiment == "neutral":
            # Deterministic strategy for neutral
            pred = normalized_text
        else:
            # Model strategy for positive/negative
            s_logits = start_logits_all[idx]
            e_logits = end_logits_all[idx]
            offsets = dataset_offsets[idx]

            pred = get_best_span(s_logits, e_logits, normalized_text, offsets)

        predictions.append(pred)

    return predictions


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # We rely on the library to handle caching and preprocessing
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)
    train_loader, val_loader, test_loader = get_loaders(
        tokenizer, load_cached_data=True
    )

    # Load raw dataframes for metric calculation and submission
    val_df = pd.read_csv(Config.VAL_FILE)
    test_df = pd.read_csv(Config.TEST_FILE)

    # 3. Model Initialization
    model = SentimentConditionedDeberta()
    model.to(device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    best_val_loss = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)

        # Validation (Loss only)
        val_loss, _, _ = eval_fn(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print("Training complete. Loading best model...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # 6. Validation Assessment
    print("Running validation inference...")
    val_preds = run_inference(model, val_loader, val_df, device)

    # Calculate Jaccard
    scores = []
    for i, row in val_df.iterrows():
        # Ground truth selected_text
        selected_text = str(row["selected_text"])
        # We compare against the normalized version of selected_text because our predictions are normalized
        # However, the metric is word-level Jaccard, which is robust to whitespace differences.
        # Ideally, we should use the raw selected_text provided in the file,
        # but our prediction is derived from normalized text.
        # The jaccard function splits on whitespace, so normalization (collapsing spaces)
        # doesn't affect the set of words, only the spacing.
        score = jaccard(val_preds[i], selected_text)
        scores.append(score)

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_df["predicted_text"] = val_preds
    val_df["jaccard"] = scores
    val_df["error"] = 1.0 - val_df["jaccard"]
    val_df["text_len"] = val_df["text"].astype(str).apply(len)

    # Correlation with text length
    corr_len = val_df[["error", "text_len"]].corr().iloc[0, 1]
    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")

    # Error by Sentiment
    print("Mean Jaccard by Sentiment:")
    print(val_df.groupby("sentiment")["jaccard"].mean())

    # 8. Submission
    THRESHOLD = 0.7043342108129372
    if final_metric > THRESHOLD:
        print("\nMetric threshold met. Generating submission...")
        test_preds = run_inference(model, test_loader, test_df, device)

        submission_df = pd.DataFrame(
            {"textID": test_df["textID"], "selected_text": test_preds}
        )

        # Ensure output is quoted correctly by pandas (default csv behavior handles strings)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
