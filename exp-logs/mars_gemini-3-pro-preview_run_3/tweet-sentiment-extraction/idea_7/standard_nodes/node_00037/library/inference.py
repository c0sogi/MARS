import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import get_loaders, get_test_loader
from library.model import TweetModel
from library.engine import train_fn, eval_fn
from library.awp import AWP


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_best_start_end_idxs(start_logits, end_logits, text, offsets):
    """
    Decodes the best span from start/end logits using Summation Decoding.
    Maximizes P_start(i) + P_end(j) subject to i <= j.
    """
    start_probs = softmax(start_logits)
    end_probs = softmax(end_logits)

    max_len = len(start_logits)
    best_score = -float("inf")
    best_span = (0, 0)

    # Iterate over all valid spans (i <= j)
    # Since tweets are short (max_len=96), O(N^2) is negligible.
    for i in range(max_len):
        for j in range(i, max_len):
            score = start_probs[i] + end_probs[j]
            if score > best_score:
                best_score = score
                best_span = (i, j)

    start_idx = best_span[0]
    end_idx = best_span[1]

    # Map token indices to character indices
    # Handle case where offsets might be (0,0) for special tokens if not filtered
    # But usually we just take what the offsets say.
    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]

    # Extract from text
    # Note: 'text' here should be the one aligned with offsets (cleaned text)
    return text[char_start:char_end]


def train_fold(fold):
    """
    Trains the model for a specific fold and saves the best checkpoint.
    """
    print(f"--- Training Fold {fold} ---")
    seed_everything(Config.seed)

    # 1. Data Loaders
    train_loader, val_loader = get_loaders(fold)

    # 2. Model Setup
    model = TweetModel(Config)
    model.to(Config.device)

    # 3. Optimizer & Scheduler
    # Adjust learning rate for the head vs backbone if needed, but simple setup per config here
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_train_steps = int(
        len(train_loader) * Config.epochs / Config.gradient_accumulation_steps
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # 4. AWP Setup
    awp = None
    if Config.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
        )

    # 5. Training Loop
    best_jaccard = -1.0
    model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.bin")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(
            train_loader, model, optimizer, Config.device, scheduler, epoch, awp
        )

        # Validate
        val_loss, (val_start_logits, val_end_logits) = eval_fn(
            val_loader, model, Config.device
        )

        # Compute Validation Jaccard
        # Note: val_loader only contains positive/negative samples (neutral excluded in get_loaders)
        val_jaccards = []
        val_dataset = val_loader.dataset

        for i in range(len(val_dataset)):
            # Get metadata
            text = val_dataset.texts[i]
            selected_text = val_dataset.selected_texts[i]
            offsets = val_dataset.offsets[i]

            # Reconstruct the cleaned text used for tokenization to align with offsets
            cleaned_text = " " + " ".join(str(text).split())

            # Decode prediction
            pred_text = get_best_start_end_idxs(
                val_start_logits[i], val_end_logits[i], cleaned_text, offsets
            )

            # Compute score
            score = jaccard(pred_text, selected_text)
            val_jaccards.append(score)

        avg_jaccard = np.mean(val_jaccards)
        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Jaccard: {avg_jaccard:.5f}"
        )

        # Save Best Model
        if avg_jaccard > best_jaccard:
            best_jaccard = avg_jaccard
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved for fold {fold}!")

    # Clean up
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    return best_jaccard


def generate_submission():
    """
    Main function to generate the submission file.
    Checks for existing models; if missing, triggers training.
    Then runs inference using the ensemble of 5 folds.
    """
    seed_everything(Config.seed)

    # 1. Check and Train Models
    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.bin")
        if not os.path.exists(model_path):
            print(f"Model for fold {fold} not found. Starting training...")
            train_fold(fold)
        else:
            print(f"Model for fold {fold} found. Skipping training.")

    # 2. Prepare for Inference
    test_loader, df_test = get_test_loader()
    n_samples = len(df_test)

    # Accumulators for ensemble logits
    final_start_logits = np.zeros((n_samples, Config.max_len))
    final_end_logits = np.zeros((n_samples, Config.max_len))

    # 3. Ensemble Inference
    print("Starting Inference...")
    for fold in range(Config.n_folds):
        print(f"Inference Fold {fold}...")
        model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.bin")

        # Load Model
        model = TweetModel(Config)
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        # Get Logits
        _, (start_logits, end_logits) = eval_fn(test_loader, model, Config.device)

        # Accumulate
        final_start_logits += start_logits
        final_end_logits += end_logits

        del model
        torch.cuda.empty_cache()

    # Average Logits
    final_start_logits /= Config.n_folds
    final_end_logits /= Config.n_folds

    # 4. Decoding
    predictions = []
    dataset = test_loader.dataset

    for i in range(n_samples):
        sentiment = df_test.iloc[i]["sentiment"]
        original_text = df_test.iloc[i]["text"]

        # Deterministic Rule for Neutral
        if sentiment == "neutral":
            predictions.append(original_text)
        else:
            # For Positive/Negative, use the model
            # Reconstruct cleaned text to match offsets
            cleaned_text = " " + " ".join(str(original_text).split())
            offsets = dataset.offsets[i]

            pred = get_best_start_end_idxs(
                final_start_logits[i], final_end_logits[i], cleaned_text, offsets
            )
            predictions.append(pred)

    # 5. Create Submission
    submission_df = pd.DataFrame(
        {"textID": df_test["textID"], "selected_text": predictions}
    )

    # Save with quoting handled by pandas (standard CSV)
    # The requirement "selected text needs to be quoted" is satisfied by standard CSV
    # if the text contains delimiters. However, to be safe and match the sample exactly:
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
