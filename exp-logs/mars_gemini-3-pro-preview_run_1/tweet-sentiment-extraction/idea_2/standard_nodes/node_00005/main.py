import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import sys
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn, infer_fn
from library.utils import jaccard


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    # Load cached data if available to optimize runtime
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = TweetModel()
    model.to(device)

    # 4. Optimizer and Scheduler Setup
    # Separate parameters to apply weight decay only to weights, not bias/LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

    # Calculate total training steps
    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # 5. Training Loop
    best_jaccard = -1.0

    for epoch in range(Config.EPOCHS):
        # Train for one epoch
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)

        # Evaluate on validation set
        val_loss, val_jaccard = eval_fn(val_loader, model, device)

        # Save best model based on Jaccard score
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Evaluation & Failure Analysis
    # Load the best model checkpoint
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    # We perform a detailed validation pass to collect per-sample metrics for failure analysis
    val_records = []

    with torch.no_grad():
        for d in val_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            texts = d["text"]
            selected_texts = d["selected_text"]
            sentiments = d["sentiment"]
            offsets = d["offsets"].cpu().numpy()

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(texts)):
                orig_text = texts[i]
                target_text = selected_texts[i]
                sentiment = sentiments[i]

                # Hybrid Inference Logic:
                # If sentiment is neutral, predict the full text.
                # Otherwise, use the model's start/end logits.
                if sentiment == "neutral":
                    pred_text = orig_text
                else:
                    start_p = start_probs[i]
                    end_p = end_probs[i]

                    # Compute score matrix P(start) + P(end)
                    score_mat = start_p[:, None] + end_p[None, :]

                    # Mask invalid spans where start > end
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    score_mat = np.where(upper_tri_mask == 1, score_mat, -100.0)

                    # Find best span
                    max_idx = np.argmax(score_mat)
                    best_start, best_end = np.unravel_index(max_idx, score_mat.shape)

                    # Map tokens back to character offsets
                    start_char = offsets[i][best_start][0]
                    end_char = offsets[i][best_end][1]

                    if best_start == best_end and start_char == 0 and end_char == 0:
                        pred_text = orig_text
                    else:
                        pred_text = orig_text[start_char:end_char]

                # Calculate Jaccard score for this sample
                score = jaccard(pred_text, target_text)

                val_records.append(
                    {
                        "text": orig_text,
                        "selected_text": target_text,
                        "prediction": pred_text,
                        "sentiment": sentiment,
                        "jaccard": score,
                        "text_len": len(orig_text),
                    }
                )

    # Create DataFrame for analysis
    df_results = pd.DataFrame(val_records)

    # Calculate Final Validation Metric
    final_metric = df_results["jaccard"].mean()
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate error magnitude (1 - Jaccard)
    df_results["error"] = 1.0 - df_results["jaccard"]

    # Calculate correlation between Error and Text Length
    corr_len = df_results["error"].corr(df_results["text_len"])
    print(f"Correlation between Error and Text Length: {corr_len}")

    # 7. Submission
    # Generate submission only if metric exceeds the specified threshold
    threshold = 0.6866348483059627

    if final_metric > threshold:
        infer_fn(test_loader, model, device)


if __name__ == "__main__":
    run()
