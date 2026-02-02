import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import warnings
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from library.config import TweetConfig
from library.utils import seed_everything, jaccard
from library.data import get_loaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn, inference_fn, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def get_optimizer_params(model, config):
    """
    Configures Layer-wise Learning Rate Decay (LLRD) for the optimizer.
    """
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = []
    param_groups = {}

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        # Determine Learning Rate
        if "backbone" not in name:
            # Head parameters get the base learning rate
            lr = config.LEARNING_RATE
        else:
            # Backbone parameters get decayed learning rate
            if "embeddings" in name:
                lr = config.LEARNING_RATE * (config.LLRD**13)
            elif "encoder.layer" in name:
                # Extract layer index for DeBERTa-v3 (0-11)
                parts = name.split(".")
                try:
                    idx = parts.index("layer")
                    layer_num = int(parts[idx + 1])
                    # Layer 11 is top (exponent 1), Layer 0 is bottom (exponent 12)
                    exponent = 12 - layer_num
                    lr = config.LEARNING_RATE * (config.LLRD**exponent)
                except:
                    lr = config.LEARNING_RATE * (config.LLRD**13)
            else:
                # Other backbone params (e.g. pooler)
                lr = config.LEARNING_RATE * (config.LLRD**13)

        # Determine Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = config.WEIGHT_DECAY

        # Group parameters by (lr, weight_decay)
        key = (lr, wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(param)

    for (lr, wd), params in param_groups.items():
        optimizer_parameters.append({"params": params, "lr": lr, "weight_decay": wd})

    return optimizer_parameters


def run():
    # 1. Configuration and Seeding
    config = TweetConfig()
    seed_everything(config.SEED)

    # 2. Data Loading
    # Utilizing cached data for speed as requested
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # 3. Model Initialization
    model = TweetModel(pretrained=True)
    model.to(config.DEVICE)

    # 4. Optimizer and Scheduler
    optimizer_parameters = get_optimizer_params(model, config)
    optimizer = optim.AdamW(optimizer_parameters)

    num_train_steps = len(train_loader) * config.EPOCHS

    if config.SCHEDULER_TYPE == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )
    else:
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
    threshold = 0.7043342108129372

    if best_jaccard > threshold:
        generate_submission(test_loader, model, config.DEVICE)
    else:
        print(
            f"Validation metric {best_jaccard} is not higher than {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run()
