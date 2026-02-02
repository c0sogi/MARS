import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import warnings
from transformers import get_linear_schedule_with_warmup

from library.config import TweetConfig
from library.utils import seed_everything, jaccard
from library.data import get_loaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn, inference_fn, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run():
    # 1. Configuration and Seeding
    config = TweetConfig()
    seed_everything(config.SEED)

    # 2. Data Loading
    # Utilizing cached data for speed as requested
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    model = TweetModel(pretrained=True)
    model.to(config.DEVICE)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    num_train_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    best_jaccard = -1.0
    best_model_path = os.path.join(config.CACHE_DIR, "best_model.bin")

    # Ensure working directory for model saving exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    for epoch in range(config.EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, config.DEVICE, scheduler)
        val_loss, val_jaccard = eval_fn(val_loader, model, config.DEVICE)

        # Save best model
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), best_model_path)

    # Print Final Validation Metric (Required Format)
    print(f"Final Validation Metric: {best_jaccard}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")

    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path))
    model.to(config.DEVICE)
    model.eval()

    # Get predictions on validation set
    # Note: val_loader excludes 'neutral' tweets, so we must filter the dataframe similarly
    val_preds = inference_fn(val_loader, model, config.DEVICE)

    # Load validation metadata and align with loader
    val_df = pd.read_csv(config.VAL_PATH)
    val_df = val_df[val_df["sentiment"] != "neutral"].reset_index(drop=True)

    # Handle Debug mode slicing if applicable
    if config.DEBUG:
        val_df = val_df.head(config.DEBUG_SIZE)

    # Safety check for alignment
    if len(val_preds) != len(val_df):
        min_len = min(len(val_preds), len(val_df))
        val_preds = val_preds[:min_len]
        val_df = val_df.iloc[:min_len]

    # Calculate Jaccard per sample
    val_df["pred_selected_text"] = val_preds
    val_df["jaccard"] = val_df.apply(
        lambda x: jaccard(x["pred_selected_text"], x["selected_text"]), axis=1
    )

    # Calculate Error Magnitude and Input Features
    val_df["error_magnitude"] = 1.0 - val_df["jaccard"]
    val_df["text_len"] = val_df["text"].astype(str).apply(len)

    # Calculate Correlation
    correlation = val_df["error_magnitude"].corr(val_df["text_len"])
    print(f"Correlation between Error Magnitude and Input Text Length: {correlation}")

    # 7. Submission Generation
    # Only generate if metric exceeds threshold
    threshold = 0.6990869212051385

    if best_jaccard > threshold:
        generate_submission(test_loader, model, config.DEVICE)
    else:
        print(
            f"Validation metric {best_jaccard} is not higher than {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run()
