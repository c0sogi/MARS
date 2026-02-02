import pandas as pd
import torch
import os
import numpy as np
import shutil
from transformers import AutoTokenizer

# Import from library
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import process_data, get_loaders, get_test_loader
from library.model import TweetModel
from library.loss import LabelSmoothingLoss, DistillationLoss
from library.engine import train_fn, eval_fn, decode_prediction
from library.inference import predict_test


def create_demo_data():
    """Creates small subsets of data for demonstration purposes."""
    print("Creating demo datasets...")

    # Read original metadata
    # We use the metadata files as they are guaranteed to exist and be clean
    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/val.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Create tiny subsets (e.g., 32 train, 16 val, 16 test)
    # This ensures the model sees enough data to form a batch but runs fast
    demo_train = train_full.head(32).copy()
    demo_val = val_full.head(16).copy()
    demo_test = test_full.head(16).copy()

    # Save to working directory
    demo_train_path = "./working/demo_train.csv"
    demo_val_path = "./working/demo_val.csv"
    demo_test_path = "./working/demo_test.csv"

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    return demo_train_path, demo_val_path, demo_test_path


def run_demo():
    # 1. Configuration Setup
    print("\n--- 1. Configuration Setup ---")
    seed_everything(42)

    # Override Config for speed and demo purposes
    Config.output_dir = "./working/demo_output/"

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.output_dir):
        shutil.rmtree(Config.output_dir)
    os.makedirs(Config.output_dir, exist_ok=True)

    # Point Config to our small demo datasets
    t_path, v_path, te_path = create_demo_data()
    Config.train_path = t_path
    Config.val_path = v_path
    Config.test_path = te_path

    # Reduce compute requirements for the demo
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.n_folds = 2  # We will only run fold 0 to save time

    print(f"Configured output directory: {Config.output_dir}")
    print(f"Device: {Config.device}")

    # 2. Data Processing & Loading
    print("\n--- 2. Data Processing & Loading ---")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Test process_data logic explicitly
    # This function loads data, tokenizes it, and saves it to cache
    print("Running process_data...")
    df, input_ids, masks, token_types, offsets, targets = process_data(
        tokenizer, mode="train"
    )

    # Assertions on processed data
    # We combined train (32) and val (16) = 48 samples
    expected_len = 32 + 16
    assert len(df) == expected_len, f"Expected {expected_len} samples, got {len(df)}"
    assert input_ids.shape == (expected_len, Config.max_len)
    assert targets.shape == (expected_len, 2)
    print("Data processing verification passed.")

    # Get DataLoaders for Fold 0
    # get_loaders internally calls process_data (which will now load from the cache we just created)
    print("Getting DataLoaders...")
    train_loader, val_loader = get_loaders(fold=0, tokenizer=tokenizer)

    # Verify DataLoader yields correct batch structure
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    assert "input_ids" in batch
    assert "start_targets" in batch
    assert batch["input_ids"].shape[0] == Config.train_batch_size
    print("DataLoader verification passed.")

    # 3. Model Initialization
    print("\n--- 3. Model Initialization ---")
    model = TweetModel()
    model.to(Config.device)

    # Verify Forward Pass
    print("Verifying forward pass...")
    with torch.no_grad():
        ids = batch["input_ids"].to(Config.device)
        mask = batch["attention_mask"].to(Config.device)
        # DeBERTa might not use token_type_ids, but we pass them if available
        types = batch["token_type_ids"].to(Config.device)

        start_logits, end_logits = model(ids, mask, types)

    assert start_logits.shape == (Config.train_batch_size, Config.max_len)
    assert end_logits.shape == (Config.train_batch_size, Config.max_len)
    print("Model forward pass verification passed.")

    # 4. Loss Function Verification
    print("\n--- 4. Loss Function Verification ---")

    # 4a. Label Smoothing Loss (Stage 1)
    crit_ls = LabelSmoothingLoss()
    loss_val = crit_ls(start_logits, batch["start_targets"].to(Config.device))
    assert not torch.isnan(loss_val)
    assert loss_val.item() > 0
    print(f"Label Smoothing Loss: {loss_val.item():.4f}")

    # 4b. Distillation Loss (Stage 2)
    crit_dist = DistillationLoss()
    # Mock teacher logits (same shape as student)
    teacher_start = torch.randn_like(start_logits)
    teacher_end = torch.randn_like(end_logits)

    loss_dist = crit_dist(
        start_logits,
        end_logits,
        teacher_start,
        teacher_end,
        batch["start_targets"].to(Config.device),
        batch["end_targets"].to(Config.device),
    )
    assert not torch.isnan(loss_dist)
    assert loss_dist.item() > 0
    print(f"Distillation Loss: {loss_dist.item():.4f}")

    # 5. Training Loop Demonstration (Stage 1)
    print("\n--- 5. Training Loop (Stage 1) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)

    # Run one epoch of training using the engine's train_fn
    print("Running training epoch...")
    train_fn(train_loader, model, optimizer, Config.device)

    # Run evaluation using the engine's eval_fn
    print("Running evaluation...")
    avg_jaccard, preds, logits_dict = eval_fn(val_loader, model, Config.device)
    print(f"Validation Jaccard: {avg_jaccard:.4f}")

    # Save model for inference test
    # We save as fold 0 because we trained on fold 0
    model_path = os.path.join(Config.output_dir, "model_fold_0.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 6. Distillation Loop Demonstration (Stage 2)
    print("\n--- 6. Training Loop (Stage 2 - Distillation) ---")
    # To demonstrate Stage 2, we need a 'soft_labels_cache'.
    # In a real run, this comes from a teacher model. Here, we mock it.
    # We use the 'logits_dict' returned by eval_fn (which contains val logits)
    # and add dummy logits for the training data to ensure coverage.

    soft_labels_cache = logits_dict.copy()

    # Add dummy logits for all text in the dataset to prevent key errors during iteration
    all_texts = df["text"].values
    for t in all_texts:
        if t not in soft_labels_cache:
            # Create dummy logits: shape (max_len,)
            # In real usage, these are numpy arrays
            soft_labels_cache[t] = (np.zeros(Config.max_len), np.zeros(Config.max_len))

    # Run training with distillation
    # The train_fn detects 'soft_labels_cache' is not None and uses DistillationLoss
    print("Running distillation training step...")
    train_fn(
        train_loader,
        model,
        optimizer,
        Config.device,
        soft_labels_cache=soft_labels_cache,
    )
    print("Distillation training step completed.")

    # 7. Inference Demonstration
    print("\n--- 7. Inference Demonstration ---")
    # predict_test generates the submission file.
    # We set load_cached_data=False to force it to process our new demo test set.

    print("Running inference on test set...")
    predict_test(load_cached_data=False)

    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print("Submission Head:")
    print(sub_df.head())

    # Validation
    assert (
        len(sub_df) == 16
    ), f"Expected 16 predictions (demo test size), got {len(sub_df)}"
    assert "textID" in sub_df.columns
    assert "selected_text" in sub_df.columns
    # Check that predictions are strings
    assert sub_df["selected_text"].dtype == object

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
