import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import importlib
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler

# Import library modules
import library.config
import library.dataset
import library.model
import library.utils
import library.trainer

importlib.reload(library.config)
importlib.reload(library.dataset)
importlib.reload(library.model)
importlib.reload(library.utils)
importlib.reload(library.trainer)

from library.config import Config
from library.dataset import get_dataloaders
from library.model import EssayScorerModel
from library.utils import set_seed, get_llrd_optimizer_params
from library.trainer import train_fn, valid_fn, inference_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # 1. Initialize Configuration
    cfg = Config()

    # Set seed for reproducibility
    set_seed(cfg.seed)

    # 2. Load DataLoaders
    # load_cached_data=True allows using existing val/test caches if they exist,
    # but will process our new subset train data from scratch.
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg, load_cached_data=True)

    # 4. Initialize Model
    print(f"Initializing Model: {cfg.model_name}")
    model = EssayScorerModel(cfg, pretrained=True)
    model.to(cfg.device)

    # 5. Optimizer and Scheduler
    # Use Layer-wise Learning Rate Decay
    optimizer_grouped_parameters = get_llrd_optimizer_params(
        model,
        base_lr=cfg.lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
        llrd_decay=cfg.llrd_decay,
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    # Calculate training steps
    num_update_steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
    max_train_steps = cfg.epochs * num_update_steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_train_steps * cfg.warmup_ratio),
        num_training_steps=max_train_steps,
    )

    scaler = GradScaler()

    # 6. Training Loop
    print("Starting Training...")
    best_score = -np.inf

    for epoch in range(cfg.epochs):
        best_score = train_fn(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            scaler,
            cfg.device,
            cfg,
            best_score,
            epoch,
        )

    # 7. Final Validation & Failure Analysis
    print("Loading best model for final validation...")
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))

    # Compute Final Validation Metric
    val_qwk, val_loss = valid_fn(model, val_loader, cfg.device, cfg)
    print(f"Final Validation Metric: {val_qwk}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()

    # Collect predictions and targets manually to align with text data
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)

            val_preds.append(outputs.view(-1).float().cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Load validation metadata to get text lengths
    # The val_loader is sequential (shuffle=False), so it matches the CSV order
    val_df = pd.read_csv(cfg.val_path)

    # Calculate Error Magnitude (Absolute difference)
    errors = np.abs(val_targets - val_preds)

    # Calculate Text Lengths (Character count)
    val_df["char_count"] = val_df["full_text"].astype(str).apply(len)
    lengths = val_df["char_count"].values

    # Calculate Correlation
    if len(errors) == len(lengths):
        correlation = np.corrcoef(errors, lengths)[0, 1]
        print(f"Correlation between Error Magnitude and Essay Length: {correlation}")
    else:
        print(
            f"Warning: Size mismatch for analysis. Errors: {len(errors)}, Lengths: {len(lengths)}"
        )

    # 8. Submission
    SUBMISSION_THRESHOLD = 0.8203336917523225

    if val_qwk > SUBMISSION_THRESHOLD:
        print(
            f"Validation metric {val_qwk} exceeds threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        ids, raw_preds = inference_fn(model, test_loader, cfg.device)

        # Post-processing: Clip to [1, 6] and round
        final_preds = np.clip(raw_preds, 1, 6)
        final_preds = np.round(final_preds).astype(int)

        submission_df = pd.DataFrame({"essay_id": ids, "score": final_preds})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(cfg.submission_path), exist_ok=True)

        submission_df.to_csv(cfg.submission_path, index=False)
        print(f"Submission saved to {cfg.submission_path}")
        print(submission_df.head())
    else:
        print(
            f"Validation metric {val_qwk} does not meet threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
