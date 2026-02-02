import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

# Import provided library components
from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import get_loaders, TweetDataset
from library.model import TweetModel
from library.awp import AWP
from library.engine import train_fn, eval_fn, loss_fn
from library.inference import get_best_start_end_idxs


def demo_pipeline():
    print("=== Starting Sentiment Extraction Library Demo ===")

    # 1. Setup & Configuration Override
    print("\n[1] Configuring environment...")
    # Modify Config for a quick demo run
    Config.working_dir = "./working/demo_run"
    Config.train_batch_size = 4
    Config.gradient_accumulation_steps = 1
    Config.epochs = 1
    Config.debug = True  # Although not fully utilized in all lib files, good practice

    # Clean up any previous demo run
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    seed_everything(Config.seed)
    device = Config.device
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.working_dir}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utilities...")
    score = jaccard("very good", "good")
    print(f"    Jaccard('very good', 'good') = {score:.4f}")
    # Intersection: {good}, Union: {very, good} -> 1/2 = 0.5
    assert abs(score - 0.5) < 1e-6, "Jaccard calculation incorrect"
    print("    Utility verification passed.")

    # 3. Data Loading (Subset for Speed)
    print("\n[3] Loading Data...")
    # We use fold 0. get_loaders handles tokenization and caching.
    # This might take a few seconds on the first run as it processes the CSV.
    full_train_loader, full_val_loader = get_loaders(fold=0)

    # Create a tiny subset for the demo (e.g., 16 samples)
    # This ensures the training loop finishes in seconds instead of minutes/hours
    subset_indices = list(range(16))
    mini_train_dataset = Subset(full_train_loader.dataset, subset_indices)

    mini_train_loader = DataLoader(
        mini_train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=False,  # Deterministic for demo
        drop_last=True,
    )

    print(f"    Full train dataset size: {len(full_train_loader.dataset)}")
    print(f"    Mini train dataset size: {len(mini_train_dataset)}")

    # Verify batch structure
    batch = next(iter(mini_train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    print("    DataLoader batch structure verified.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = TweetModel(Config)
    model.to(device)
    model.train()

    # Verify Forward Pass
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    start_logits, end_logits = model(input_ids, mask)
    print(f"    Logits Shape: {start_logits.shape}")

    assert start_logits.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), f"Expected start_logits shape {(Config.train_batch_size, Config.max_len)}, got {start_logits.shape}"
    assert end_logits.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), f"Expected end_logits shape {(Config.train_batch_size, Config.max_len)}, got {end_logits.shape}"
    print("    Model forward pass verified.")

    # 5. AWP Verification
    print("\n[5] Verifying Adversarial Weight Perturbation (AWP)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # AWP requires gradients to exist, so we run a backward pass first
    start_pos = batch["start_positions"].to(device)
    end_pos = batch["end_positions"].to(device)
    loss = loss_fn(start_logits, end_logits, start_pos, end_pos)
    loss.backward()

    # Initialize AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=1e-4,
        adv_eps=1e-4,
        start_epoch=0,  # Force start immediately for demo
    )

    # Save a copy of a specific parameter to check for changes
    # We pick a parameter from the classifier head as it's small and easy to check
    param_name = "classifier.weight"
    original_weight = model.classifier.weight.data.clone()

    # Perform attack step
    awp.attack_step()

    perturbed_weight = model.classifier.weight.data

    # Check if weights changed
    diff = torch.norm(original_weight - perturbed_weight).item()
    print(f"    Weight perturbation magnitude: {diff:.8f}")
    assert diff > 0, "AWP attack step did not perturb weights!"

    # Restore weights
    awp.restore()
    restored_weight = model.classifier.weight.data

    # Check if weights are restored
    restore_diff = torch.norm(original_weight - restored_weight).item()
    print(f"    Restoration difference: {restore_diff:.8f}")
    assert restore_diff < 1e-8, "AWP restore failed to return to original weights!"

    # Clear gradients for next step
    optimizer.zero_grad()
    print("    AWP logic verified.")

    # 6. Training Loop Demo
    print("\n[6] Running Training Loop (Mini-Batch)...")
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

    # Run train_fn for 1 epoch on the mini_loader
    # This uses the engine.py logic
    avg_loss = train_fn(
        mini_train_loader,
        model,
        optimizer,
        device,
        scheduler,
        epoch=1,  # Ensure AWP logic inside train_fn is triggered if configured
        awp=awp,
    )
    print(f"    Training finished. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss returned NaN"

    # 7. Inference & Decoding Logic
    print("\n[7] Verifying Decoding Logic...")
    # Create synthetic logits:
    #   Text: "hello world"
    #   Tokens: [CLS, hello, world, SEP] -> indices 0, 1, 2, 3
    #   Target: "world" -> start=2, end=2

    seq_len = 5
    syn_start_logits = np.array([-10, -10, 10, -10, -10])  # High score at index 2
    syn_end_logits = np.array([-10, -10, 10, -10, -10])  # High score at index 2

    # Mock offsets: (start_char, end_char)
    # " hello world" -> " " (0,0), "hello" (1,6), "world" (7,12)
    # Note: Tokenizer adds special tokens. Let's assume indices 1 and 2 are words.
    syn_text = " hello world"
    syn_offsets = [(0, 0), (1, 6), (7, 12), (0, 0), (0, 0)]

    prediction = get_best_start_end_idxs(
        syn_start_logits, syn_end_logits, syn_text, syn_offsets
    )

    print(f"    Text: '{syn_text}'")
    print(f"    Predicted Span: '{prediction}'")

    assert (
        prediction == "world"
    ), f"Decoding failed. Expected 'world', got '{prediction}'"
    print("    Decoding logic verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    try:
        demo_pipeline()
    except AssertionError as e:
        print(f"\n!!! Validation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
