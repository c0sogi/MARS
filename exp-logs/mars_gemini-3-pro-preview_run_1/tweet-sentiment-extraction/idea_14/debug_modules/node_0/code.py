import sys
import os
import shutil
import pandas as pd
import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure the current directory is in path to import library modules
sys.path.append(".")

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_optimizer_params
import library.data  # Import module to patch tqdm
from library.data import get_loaders, get_test_loader
from library.model import TweetModel
from library.loss import SoftTargetKLLoss, RDropLoss
from library.engine import train_fn, eval_fn

# Silence tqdm for cleaner output
library.data.tqdm = lambda x, **kwargs: x

# Silence transformers logging
transformers.logging.set_verbosity_error()


def run_demo():
    print(">>> Starting Sentiment Extraction Demo...")

    # ====================================================
    # 1. Setup & Configuration Overrides
    # ====================================================
    # Set seed for reproducibility
    seed_everything(42)

    # Define a temporary working directory for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # Override Config parameters for speed
    Config.WORKING_DIR = demo_dir
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    # We will force re-processing of data to use our small subsets

    print(f">>> Configured temporary working directory: {Config.WORKING_DIR}")

    # Create small subsets of the metadata to speed up execution
    print(">>> Creating metadata subsets (50 Train, 20 Val, 20 Test)...")

    # Load original metadata
    full_train = pd.read_csv(Config.TRAIN_META_PATH)
    full_val = pd.read_csv(Config.VAL_META_PATH)
    full_test = pd.read_csv(Config.TEST_META_PATH)

    # Select non-neutral samples for train/val (as the pipeline filters neutrals)
    # This ensures our subset isn't empty after filtering
    train_subset = full_train[full_train["sentiment"] != "neutral"].head(50)
    val_subset = full_val[full_val["sentiment"] != "neutral"].head(20)
    test_subset = full_test.head(20)  # Test set includes neutrals

    # Save subsets to the demo directory
    demo_train_path = os.path.join(demo_dir, "demo_train.csv")
    demo_val_path = os.path.join(demo_dir, "demo_val.csv")
    demo_test_path = os.path.join(demo_dir, "demo_test.csv")

    train_subset.to_csv(demo_train_path, index=False)
    val_subset.to_csv(demo_val_path, index=False)
    test_subset.to_csv(demo_test_path, index=False)

    # Point Config to the new subset files
    Config.TRAIN_META_PATH = demo_train_path
    Config.VAL_META_PATH = demo_val_path
    Config.TEST_META_PATH = demo_test_path

    # ====================================================
    # 2. Data Loading
    # ====================================================
    print("\n>>> Initializing Tokenizer and DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # load_cached_data=False ensures we process the new CSVs we just created
    train_loader, val_loader = get_loaders(tokenizer, load_cached_data=False)

    # Verification: Check batch structure and shapes
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")

    # Assertions to verify data integrity
    assert "input_ids" in batch
    assert "start_targets" in batch
    assert batch["input_ids"].shape == (Config.train_batch_size, Config.max_len)
    assert batch["start_targets"].shape == (Config.train_batch_size, Config.max_len)

    # Verify targets are valid probability distributions (sum <= 1.0 + epsilon)
    # Sum might be 0 if target text not found, but usually ~1.0
    target_sums = batch["start_targets"].sum(dim=1)
    assert torch.all(target_sums <= 1.01), "Target distributions sum to > 1"
    print("Data validation passed.")

    # ====================================================
    # 3. Model Initialization
    # ====================================================
    print("\n>>> Initializing Model (DeBERTa-v3-large)...")
    device = Config.device
    model = TweetModel(Config)
    model.to(device)

    # Verification: Forward pass
    print("Running dummy forward pass...")
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask)

    # Check output shapes: (batch_size, seq_len)
    assert start_logits.shape == (Config.train_batch_size, Config.max_len)
    assert end_logits.shape == (Config.train_batch_size, Config.max_len)
    print(f"Model output shape verified: {start_logits.shape}")

    # ====================================================
    # 4. Loss Functions
    # ====================================================
    print("\n>>> Testing Loss Functions...")
    criterion_task = SoftTargetKLLoss()
    criterion_rdrop = RDropLoss()

    start_targets = batch["start_targets"].to(device)

    # Calculate Task Loss
    loss_val = criterion_task(start_logits, start_targets)
    print(f"Task Loss: {loss_val.item():.4f}")
    assert not torch.isnan(loss_val), "Loss is NaN"

    # Calculate R-Drop Loss (Self-consistency)
    # Loss between identical logits should be effectively 0
    rdrop_val = criterion_rdrop(start_logits, start_logits)
    print(f"R-Drop Self-Loss: {rdrop_val.item():.6f}")
    assert rdrop_val.item() < 1e-5, "R-Drop loss for identical inputs should be ~0"

    # ====================================================
    # 5. Training Loop Simulation
    # ====================================================
    print("\n>>> Simulating Training (1 Epoch on subset)...")

    # Setup Optimizer with Layer-wise Learning Rate Decay
    optimizer_params = get_optimizer_params(model, encoder_lr=2e-5, decoder_lr=1e-4)
    optimizer = torch.optim.AdamW(optimizer_params)

    # Setup Scheduler
    num_training_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Run Training Function
    avg_train_loss = train_fn(
        train_loader,
        model,
        optimizer,
        device,
        scheduler,
        criterion_task,
        criterion_rdrop,
        Config,
    )
    print(f"Epoch 1 Train Loss: {avg_train_loss:.4f}")

    # ====================================================
    # 6. Evaluation Simulation
    # ====================================================
    print("\n>>> Simulating Evaluation...")
    val_loss, val_jaccard = eval_fn(val_loader, model, device, criterion_task, Config)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Jaccard: {val_jaccard:.4f}")

    # ====================================================
    # 7. Inference / Submission
    # ====================================================
    print("\n>>> Running Inference on Test Subset...")

    # Get Test Loader (targets are None)
    test_loader = get_test_loader(tokenizer)

    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            offsets = batch["offsets"].numpy()
            texts = batch["text"]

            # Forward pass
            start_logits, end_logits = model(input_ids, mask)

            # Convert logits to probabilities
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            # Decode probabilities to text spans
            for i in range(len(texts)):
                start_p = start_probs[i]
                end_p = end_probs[i]
                offset = offsets[i]
                original_text = texts[i]

                # Maximize joint probability P(start) * P(end) subject to start <= end
                score_mat = np.outer(start_p, end_p)
                score_mat = np.triu(
                    score_mat
                )  # Zero out invalid spans where end < start

                best_idx = np.argmax(score_mat)
                best_start_idx, best_end_idx = np.unravel_index(
                    best_idx, score_mat.shape
                )

                # Map token indices back to character indices using offsets
                if best_start_idx >= len(offset) or best_end_idx >= len(offset):
                    pred_text = original_text
                else:
                    char_start = offset[best_start_idx][0]
                    char_end = offset[best_end_idx][1]

                    # Handle [CLS] token or empty spans
                    if char_start == 0 and char_end == 0:
                        pred_text = original_text
                    else:
                        pred_text = original_text[char_start:char_end]

                predictions.append(pred_text)

    # Format submission
    # Note: We use test_subset.textID because the loader preserves order
    submission_df = pd.DataFrame(
        {"textID": test_subset["textID"], "selected_text": predictions}
    )

    print("\nSample Predictions:")
    print(submission_df.head())

    # Verify submission format
    assert "textID" in submission_df.columns
    assert "selected_text" in submission_df.columns
    assert len(submission_df) == len(test_subset)

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
