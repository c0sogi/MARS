import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.dataset import get_loaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn
from library.utils import seed_everything, jaccard, get_selected_text


def run_fold(fold):
    """
    Trains and validates the model for a single fold.
    """
    print(f"\n{'='*20} Fold: {fold} {'='*20}")

    # 1. Setup Data
    train_loader, val_loader = get_loaders(
        fold, load_cached_data=True, debug=Config.DEBUG
    )

    # 2. Setup Model
    device = Config.DEVICE
    model = TweetModel(Config)
    model.to(device)

    # 3. Setup Optimizer
    # Apply weight decay to all parameters except bias and LayerNorm weights
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_parameters, lr=Config.LEARNING_RATE)

    # 4. Setup Scheduler
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # 5. Training Loop
    best_jaccard = 0.0
    best_model_path = os.path.join(Config.MODEL_OUTPUT_DIR, f"model_fold_{fold}.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)

        # Evaluate (Get logits)
        start_preds, end_preds, val_loss = eval_fn(val_loader, model, device)

        # Calculate Jaccard Score
        # We need to reconstruct the texts from the validation loader to compare
        # Since shuffle=False for validation, we can iterate to collect metadata
        val_texts = []
        val_sentiments = []
        val_selected_texts = []
        val_offsets = []

        # Collect metadata from validation set
        # Note: We iterate the loader again. This is cheap compared to model inference.
        for batch in val_loader:
            val_texts.extend(batch["text"])
            val_sentiments.extend(batch["sentiment"])
            val_selected_texts.extend(batch["selected_text"])
            val_offsets.extend(batch["offsets"].numpy())

        jaccard_scores = []

        for i in range(len(val_texts)):
            text = val_texts[i]
            sentiment = val_sentiments[i]
            selected_text = val_selected_texts[i]
            offset = val_offsets[i]

            start_logits = start_preds[i]
            end_logits = end_preds[i]

            # Heuristic: Neutral sentiment -> Full text
            if sentiment == "neutral" and Config.NEUTRAL_FULL_TEXT:
                pred_text = text
            else:
                # Decoding: Find (start, end) that maximizes start_logits + end_logits
                # subject to start <= end
                # We can do this efficiently with numpy

                # Create a sum matrix: (seq_len, seq_len)
                sum_matrix = start_logits[:, None] + end_logits[None, :]

                # Mask out the lower triangle (where start > end)
                # We use a large negative number for invalid positions
                seq_len = len(start_logits)
                mask = np.triu(np.ones((seq_len, seq_len)), k=0)
                sum_matrix = sum_matrix * mask + (1 - mask) * -1e9

                # Find the indices of the maximum value
                flat_idx = np.argmax(sum_matrix)
                start_idx = flat_idx // seq_len
                end_idx = flat_idx % seq_len

                pred_text = get_selected_text(text, start_idx, end_idx, offset)

            score = jaccard(pred_text, selected_text)
            jaccard_scores.append(score)

        avg_jaccard = np.mean(jaccard_scores)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Jaccard: {avg_jaccard}"
        )

        # Save Best Model
        if avg_jaccard > best_jaccard:
            best_jaccard = avg_jaccard
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Model Saved! New Best Jaccard: {best_jaccard}")

    # Cleanup to save memory
    del model, optimizer, scheduler, train_loader, val_loader, start_preds, end_preds
    torch.cuda.empty_cache()

    return best_jaccard


def run_training():
    """
    Orchestrates the training across all folds.
    """
    seed_everything(Config.SEED)

    # Ensure output directory exists
    os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)

    fold_scores = []

    for fold in range(Config.N_FOLDS):
        score = run_fold(fold)
        fold_scores.append(score)

    print(f"\n{'='*20} Training Complete {'='*20}")
    print("Fold Scores:", fold_scores)
    print(f"Average Jaccard: {np.mean(fold_scores)}")


if __name__ == "__main__":
    run_training()
