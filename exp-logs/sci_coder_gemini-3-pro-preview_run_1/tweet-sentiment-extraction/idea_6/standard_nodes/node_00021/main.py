import sys
import os
import pandas as pd
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Suppress tqdm output to comply with "No progress bars" requirement
import tqdm


class SilentTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable else []

    def __iter__(self):
        return iter(self.iterable)

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def set_description(self, *args, **kwargs):
        pass

    @classmethod
    def write(cls, *args, **kwargs):
        pass


tqdm.tqdm = SilentTqdm

# Import library modules after suppressing tqdm
from library.config import Config
from library.utils import seed_everything, jaccard, process_text
from library.data import get_loaders
from library.model import SentimentModel
from library.engine import fit, predict, decode_prediction


def main():
    # Configuration Overrides
    Config.epochs = 5

    # Setup
    seed_everything(Config.seed)
    warnings.filterwarnings("ignore")

    # Data Loading
    # load_cached_data=True allows skipping preprocessing if artifacts exist
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_loader, val_loader, test_loader = get_loaders(
        tokenizer, load_cached_data=True
    )

    # Model Initialization
    model = SentimentModel(Config)
    model.to(Config.device)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Training Loop
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.device,
        epochs=Config.epochs,
        patience=Config.early_stopping_patience,
        model_save_path=Config.model_save_path,
    )

    # =========================================================================
    # Validation & Failure Analysis
    # =========================================================================
    model.eval()

    val_preds = []
    val_targets = []
    val_sentiments = []
    val_texts = []

    # 1. Active Validation Set (Positive/Negative) - Model Inference
    # Note: val_loader only contains non-neutral tweets because Config.filter_neutral=True
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            texts = batch["text"]
            selected_texts = batch["selected_text"]
            sentiments = batch["sentiment"]
            offsets = batch["offsets"].numpy()

            s_logits, e_logits = model(input_ids, attention_mask)

            for i in range(len(texts)):
                pred = decode_prediction(
                    s_logits[i], e_logits[i], texts[i], offsets[i], sentiments[i]
                )
                val_preds.append(pred)
                val_targets.append(selected_texts[i])
                val_sentiments.append(sentiments[i])
                val_texts.append(texts[i])

    # 2. Neutral Validation Set - Heuristic Inference
    # We must evaluate on the FULL validation set. Since val_loader filters neutrals,
    # we load the metadata manually to process the neutral samples.
    df_val_full = pd.read_csv(Config.val_path)
    df_val_neutral = df_val_full[df_val_full["sentiment"] == "neutral"]

    for _, row in df_val_neutral.iterrows():
        # Apply normalization to match the pipeline's text state
        norm_text = process_text(row["text"])
        norm_selected = process_text(row["selected_text"])

        # Prediction for neutral is the text itself (Identity Mapping)
        val_preds.append(norm_text)
        val_targets.append(norm_selected)
        val_sentiments.append("neutral")
        val_texts.append(norm_text)

    # Calculate Metrics
    scores = [jaccard(t, p) for t, p in zip(val_targets, val_preds)]
    final_metric = np.mean(scores)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    errors = [1.0 - s for s in scores]
    text_lens = [len(str(t)) for t in val_texts]

    # Map sentiment to numeric for correlation
    sent_map = {"negative": -1, "neutral": 0, "positive": 1}
    sent_nums = [sent_map[s] for s in val_sentiments]

    if len(errors) > 1:
        corr_len = np.corrcoef(errors, text_lens)[0, 1]
        corr_sent = np.corrcoef(errors, sent_nums)[0, 1]
    else:
        corr_len = 0.0
        corr_sent = 0.0

    print(f"Correlation (Error vs Text Length): {corr_len}")
    print(f"Correlation (Error vs Sentiment): {corr_sent}")

    # =========================================================================
    # Submission
    # =========================================================================
    threshold = 0.7043342108129372
    if final_metric > threshold:
        predict(test_loader, model, Config.device, Config.submission_path)


if __name__ == "__main__":
    main()
