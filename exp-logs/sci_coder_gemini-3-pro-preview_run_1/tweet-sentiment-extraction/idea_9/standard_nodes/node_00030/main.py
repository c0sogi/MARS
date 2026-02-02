import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, normalize_text, jaccard
from library.data import get_loaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn, generate_submission


def run():
    # 1. Setup and Configuration
    # Adjust Config for a fast baseline execution on A100
    Config.EPOCHS = 5
    Config.TRAIN_BATCH_SIZE = 32
    Config.VALID_BATCH_SIZE = 64

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    # Use cached data if available to speed up the process
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=True,
        batch_size=Config.TRAIN_BATCH_SIZE,
        val_batch_size=Config.VALID_BATCH_SIZE,
    )

    # 3. Model Initialization
    model = TweetModel()
    model.to(device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    best_jaccard = 0.0
    model_path = Config.MODEL_SAVE_PATH

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_loss, val_jaccard = eval_fn(val_loader, model, device)

        # Save best model
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), model_path)

    # 6. Final Validation & Failure Analysis
    # Load the best model for analysis
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Perform detailed inference on validation set for analysis
    val_records = []

    with torch.no_grad():
        for data in val_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            sentiment = data["sentiment"]
            texts = data["text"]
            selected_texts = data["selected_text"]
            offsets = data["offsets"].cpu().numpy()

            logits = model(input_ids, attention_mask)
            start_logits = logits[:, :, 0].cpu().numpy()
            end_logits = logits[:, :, 1].cpu().numpy()

            for i in range(len(texts)):
                orig_text = texts[i]
                current_sentiment = sentiment[i]
                true_selected = selected_texts[i]

                if current_sentiment == "neutral":
                    pred_text = orig_text
                else:
                    start_pred = start_logits[i]
                    end_pred = end_logits[i]

                    scores = start_pred[:, None] + end_pred[None, :]
                    upper_tri_mask = np.triu(np.ones_like(scores))
                    scores = np.where(upper_tri_mask == 1, scores, -np.inf)

                    max_idx = np.argmax(scores)
                    start_idx, end_idx = np.unravel_index(max_idx, scores.shape)

                    norm_text = normalize_text(orig_text)
                    current_offsets = offsets[i]

                    if start_idx < len(current_offsets) and end_idx < len(
                        current_offsets
                    ):
                        start_char = current_offsets[start_idx][0]
                        end_char = current_offsets[end_idx][1]
                        if end_char > start_char and end_char <= len(norm_text):
                            pred_text = norm_text[start_char:end_char]
                        else:
                            pred_text = norm_text
                    else:
                        pred_text = norm_text

                score = jaccard(pred_text, true_selected)

                val_records.append(
                    {
                        "text_len": len(orig_text),
                        "sentiment": current_sentiment,
                        "jaccard": score,
                        "error": 1.0 - score,
                    }
                )

    df_analysis = pd.DataFrame(val_records)

    # Calculate and print Final Validation Metric
    final_metric = df_analysis["jaccard"].mean()
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    # Map sentiment to numeric for correlation (neg=0, neu=1, pos=2)
    sent_map = {"negative": 0, "neutral": 1, "positive": 2}
    df_analysis["sentiment_code"] = df_analysis["sentiment"].map(sent_map)

    corr_len = df_analysis["error"].corr(df_analysis["text_len"])
    corr_sent = df_analysis["error"].corr(df_analysis["sentiment_code"])

    print("Failure Analysis Report:")
    print(f"Correlation (Error vs Text Length): {corr_len}")
    print(f"Correlation (Error vs Sentiment): {corr_sent}")

    # 7. Conditional Submission
    threshold = 0.7043342108129372
    if final_metric > threshold:
        generate_submission(test_loader, model, device)


if __name__ == "__main__":
    run()
