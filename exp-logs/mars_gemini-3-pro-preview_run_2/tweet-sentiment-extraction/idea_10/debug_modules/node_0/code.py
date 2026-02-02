import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, jaccard, AWP
from library.data import get_data_loaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn, loss_fn


def run_demo():
    print("=== Starting Tweet Sentiment Extraction Demo ===\n")

    # 1. Setup Configuration
    # Enable debug mode to use only 100 samples for speed.
    # Reduce epochs to 1 for demonstration.
    config = Config(debug=True, epochs=1, train_batch_size=4)

    # Override num_workers to 0 for simpler debugging/execution in this script
    config.num_workers = 0

    print(f"Configuration: Debug={config.debug}, Device={config.device}")
    seed_everything(config.seed)

    # 2. Data Loading
    print("\n[Step 1] Loading and Processing Data...")
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Verify DataLoaders
    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches:   {len(val_loader)}")

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    start_labels = batch["start_labels"]

    print(f"Sample Batch Shapes:")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Start Labels: {start_labels.shape}")

    # Assertion: Check if batch size matches config (or remainder)
    assert input_ids.shape[0] <= config.train_batch_size
    # Assertion: Check if input_ids and attention_mask have same shape
    assert input_ids.shape == attention_mask.shape

    # 3. Model Initialization
    print("\n[Step 2] Initializing Model...")
    model = TweetModel(config)
    model.to(config.device)

    # Verify Forward Pass
    print("Verifying Forward Pass...")
    model.eval()
    with torch.no_grad():
        # Move inputs to device
        b_input_ids = input_ids.to(config.device)
        b_mask = attention_mask.to(config.device)

        start_logits, end_logits = model(b_input_ids, b_mask)

    print(f"  Logits Shape: {start_logits.shape}")

    # Assertion: Output shape should be (Batch, Seq_Len)
    assert start_logits.shape == b_input_ids.shape
    assert end_logits.shape == b_input_ids.shape
    print("Forward pass verification successful.")

    # 4. AWP (Adversarial Weight Perturbation) Verification
    print("\n[Step 3] Verifying AWP Logic...")
    # To verify AWP, we need gradients. We'll do a dummy backward pass.
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    awp = AWP(model, optimizer, adv_eps=1e-2)

    # Pick a parameter to monitor (e.g., the weight of the final fc layer)
    param_name = "fc.weight"
    original_weight = model.fc.weight.clone().detach()

    # Dummy loss and backward
    start_logits, end_logits = model(b_input_ids, b_mask)
    loss = loss_fn(
        start_logits,
        end_logits,
        start_labels.to(config.device),
        batch["end_labels"].to(config.device),
        config,
    )
    loss.backward()

    # Apply Attack
    awp.attack()
    perturbed_weight = model.fc.weight.clone().detach()

    # Check if weights changed
    diff = torch.norm(perturbed_weight - original_weight).item()
    print(f"  Weight perturbation magnitude: {diff:.6f}")
    if diff == 0.0:
        print(
            "  Warning: Weights did not change. This might happen if gradients are zero."
        )
    else:
        print("  AWP Attack successful (weights perturbed).")

    # Restore
    awp.restore()
    restored_weight = model.fc.weight.clone().detach()

    # Assertion: Weights should be restored exactly
    assert torch.allclose(original_weight, restored_weight), "AWP Restore failed!"
    print("  AWP Restore successful.")

    # Clear gradients
    optimizer.zero_grad()

    # 5. Training Loop Demonstration
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    # Setup Scheduler
    num_train_steps = int(len(train_loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Run Train Function
    avg_train_loss = train_fn(
        train_loader, model, optimizer, config.device, scheduler, config, epoch=0
    )
    print(f"  Average Train Loss: {avg_train_loss:.4f}")

    # 6. Evaluation Demonstration
    print("\n[Step 5] Running Evaluation Loop...")
    val_loss, val_jaccard = eval_fn(val_loader, model, config.device, config)

    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Jaccard: {val_jaccard:.4f}")

    # Assertion: Jaccard should be between 0 and 1
    assert 0.0 <= val_jaccard <= 1.0, "Jaccard score out of bounds"

    # 7. Utility Verification
    print("\n[Step 6] Verifying Utilities...")
    s1 = "hello world"
    s2 = "hello"
    score = jaccard(s1, s2)
    print(f"  Jaccard('{s1}', '{s2}') = {score}")
    assert score == 0.5, "Jaccard calculation incorrect"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure no warnings clutter the output
    import warnings

    warnings.filterwarnings("ignore")

    run_demo()
