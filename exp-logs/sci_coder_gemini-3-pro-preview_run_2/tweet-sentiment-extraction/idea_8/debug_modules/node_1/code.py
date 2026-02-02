import os
import shutil
import torch
import warnings
import numpy as np
import pandas as pd
import logging
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, logging as hf_logging

# Import from the provided library
from library.config import Config
from library.data import get_loaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn
from library.utils import seed_everything


# ==========================================
# 1. Setup and Configuration Overrides
# ==========================================
def setup_environment():
    # Suppress warnings and logs for cleaner output
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Override Config for a fast demonstration (Debug Mode)
    print("Configuring environment for DEMO run...")
    Config.DEBUG = True  # Uses top 100 train, 50 val, 50 test samples
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Use a specific directory for this demo to avoid overwriting existing work
    Config.WORKING_DIR = "./working/demo_execution/"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)


# ==========================================
# 2. Inference Logic (Custom for Test Set)
# ==========================================
def predict_test_set(model, test_loader, device):
    """
    Generates predictions for the test set.
    Note: library.engine.eval_fn requires targets (selected_text) which test set lacks.
    We reimplement the decoding logic here.
    """
    model.eval()
    predictions = []

    print("Running inference on test set...")
    with torch.no_grad():
        for data in test_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)

            # Metadata
            offsets = data["offsets"].cpu().numpy()
            texts = data["text"]
            sentiments = data["sentiment"]
            ids = data["textID"]

            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            for i in range(len(texts)):
                text = texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]
                text_id = ids[i]

                # Neutral Heuristic: Predict full text
                if sentiment == "neutral":
                    pred_text = text
                else:
                    start_l = start_logits[i]
                    end_l = end_logits[i]

                    # Score matrix: start_logit + end_logit
                    score_mat = start_l[:, None] + end_l[None, :]

                    # Enforce start <= end
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    score_mat = np.where(upper_tri_mask == 1, score_mat, -np.inf)

                    # Get best indices
                    best_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
                    idx_start, idx_end = best_idx

                    # extract text based on offsets
                    char_start = offset[idx_start][0]
                    char_end = offset[idx_end][1]

                    if char_start == 0 and char_end == 0:
                        pred_text = text
                    else:
                        pred_text = text[char_start:char_end]

                predictions.append({"textID": text_id, "selected_text": pred_text})

    return pd.DataFrame(predictions)


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    setup_environment()

    # --------------------------------------
    # A. Data Loading
    # --------------------------------------
    print("\n--- Data Loading ---")
    # load_cached_data=False forces processing, ensuring we test the tokenizer logic
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Verify DataLoaders
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "start_idx",
        "end_idx",
        "text",
        "sentiment",
    ]
    for key in required_keys:
        if key not in sample_batch:
            raise AssertionError(f"Batch missing key: {key}")

    print("Batch structure verified.")
    print(f"Sample Input Shape: {sample_batch['input_ids'].shape}")

    # --------------------------------------
    # B. Model Initialization
    # --------------------------------------
    print("\n--- Model Initialization ---")
    model = TweetModel()
    model.to(Config.DEVICE)
    print("Model loaded and moved to device.")

    # --------------------------------------
    # C. Training Loop
    # --------------------------------------
    print("\n--- Starting Training (1 Epoch - Debug) ---")
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # Train
    avg_loss = train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)
    print(f"Epoch 1 Loss: {avg_loss:.4f}")

    if not np.isfinite(avg_loss):
        raise AssertionError("Training loss is not finite (NaN or Inf).")

    # --------------------------------------
    # D. Validation
    # --------------------------------------
    print("\n--- Starting Validation ---")
    jaccard_score = eval_fn(val_loader, model, Config.DEVICE)
    print(f"Validation Jaccard Score: {jaccard_score:.4f}")

    if not (0.0 <= jaccard_score <= 1.0):
        raise AssertionError("Jaccard score out of bounds [0, 1].")

    # --------------------------------------
    # E. Inference & Submission
    # --------------------------------------
    print("\n--- Generating Submission ---")
    submission_df = predict_test_set(model, test_loader, Config.DEVICE)

    # Verify Submission Format
    print("Sample Predictions:")
    print(submission_df.head())

    if len(submission_df) != len(
        pd.read_csv(Config.TEST_META_PATH).head(50)
    ):  # 50 because DEBUG=True
        raise AssertionError("Submission row count does not match test set size.")

    # Save Submission
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    print("\nDemo execution completed successfully.")
