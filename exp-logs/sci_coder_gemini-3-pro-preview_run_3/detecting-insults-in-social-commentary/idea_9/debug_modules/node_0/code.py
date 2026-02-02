import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup

# Import provided library modules
from library.config import ModelConfig
from library.utils import seed_everything, decode_text
from library.data import load_processed_data, get_dataloaders, create_augmented_dataset
from library.model import InsultModel, AWP
from library.engine import train_one_epoch, valid_fn, inference_fn


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup & Configuration Override for Speed
    print("\n[1] Configuring environment for speed...")
    seed_everything(42)

    # Override Config for the demo
    ModelConfig.debug = True  # Uses only 50 samples
    ModelConfig.epochs = 1
    ModelConfig.train_batch_size = 4
    ModelConfig.valid_batch_size = 8
    ModelConfig.accumulation_steps = 1
    # Use a tiny model to ensure this runs in seconds
    demo_model_name = "prajjwal1/bert-tiny"
    ModelConfig.backbones = [demo_model_name]
    # Disable freezing for the tiny model (it only has 2 layers) to avoid index errors
    ModelConfig.freeze_layers = 0
    # Enable AWP immediately to test it
    ModelConfig.use_awp = True
    ModelConfig.awp_start_epoch = 0

    device = ModelConfig.device
    print(f"    Device: {device}")
    print(f"    Model: {demo_model_name}")
    print(f"    Debug Mode: {ModelConfig.debug}")

    # 2. Data Loading
    print("\n[2] Loading processed data...")
    train_df, val_df, test_df = load_processed_data(load_cached_data=False, debug=True)

    # Verification
    assert (
        len(train_df) <= ModelConfig.debug_sample_size
    ), "Train DF size mismatch for debug mode"
    assert (
        "Comment" in train_df.columns and "Insult" in train_df.columns
    ), "Train DF missing columns"
    print(f"    Train shape: {train_df.shape}")
    print(f"    Val shape:   {val_df.shape}")
    print(f"    Test shape:  {test_df.shape}")

    # 3. DataLoader Creation
    print("\n[3] Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, tokenizer_name=demo_model_name
    )

    # Verification
    batch = next(iter(train_loader))
    assert "input_ids" in batch, "Batch missing input_ids"
    assert "attention_mask" in batch, "Batch missing attention_mask"
    assert "target" in batch, "Batch missing target"
    assert (
        batch["input_ids"].shape[0] == ModelConfig.train_batch_size
    ), "Batch size mismatch"
    print("    DataLoaders created and batch structure verified.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = InsultModel(demo_model_name, config=ModelConfig)
    model.to(device)

    # Verification: Forward pass
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids, mask)

    assert logits.shape == (
        ModelConfig.train_batch_size,
        1,
    ), f"Logit shape mismatch: {logits.shape}"
    print("    Model initialized and forward pass verified.")

    # 5. AWP Logic Verification
    print("\n[5] Verifying Adversarial Weight Perturbation (AWP)...")
    optimizer = AdamW(model.parameters(), lr=ModelConfig.learning_rate)
    awp = AWP(
        model, optimizer, adv_lr=0.1, adv_eps=0.1
    )  # High LR to ensure visible change

    # Get a reference to a specific weight
    param_name = list(model.named_parameters())[0][0]
    original_weight = list(model.named_parameters())[0][1].data.clone()

    # Simulate gradients
    loss = logits.mean()
    loss.backward()

    # Attack
    awp.attack()
    perturbed_weight = list(model.named_parameters())[0][1].data

    # Check if weights changed
    diff = torch.norm(perturbed_weight - original_weight).item()
    assert diff > 0, "AWP attack did not perturb weights"
    print(f"    AWP Attack confirmed. Weight difference norm: {diff:.6f}")

    # Restore
    awp.restore()
    restored_weight = list(model.named_parameters())[0][1].data

    # Check if weights restored
    restore_diff = torch.norm(restored_weight - original_weight).item()
    assert restore_diff < 1e-6, "AWP restore failed"
    print("    AWP Restore confirmed.")

    # Clear gradients
    optimizer.zero_grad()

    # 6. Training Loop Execution
    print("\n[6] Running Training Loop (1 Epoch)...")
    num_train_steps = len(train_loader) * ModelConfig.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    avg_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, device, epoch=0
    )

    assert isinstance(avg_loss, float), "train_one_epoch did not return a float loss"
    print(f"    Training complete. Average Loss: {avg_loss:.4f}")

    # 7. Validation Execution
    print("\n[7] Running Validation...")
    val_loss, val_auc = valid_fn(model, val_loader, device)

    assert 0 <= val_auc <= 1, f"Invalid AUC score: {val_auc}"
    print(f"    Validation complete. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 8. Inference Execution
    print("\n[8] Running Inference on Test Set...")
    test_probs = inference_fn(model, test_loader, device)

    assert len(test_probs) == len(test_df), "Prediction count mismatch"
    assert (test_probs >= 0).all() and (
        test_probs <= 1
    ).all(), "Predictions out of probability range"
    print(f"    Inference complete. Generated {len(test_probs)} predictions.")

    # 9. Augmentation Verification
    print("\n[9] Verifying Data Augmentation (Pseudo-Labeling)...")
    # Create dummy probabilities to force pseudo-labeling
    # Set first half to 0.99 (Insult) and second half to 0.01 (Neutral)
    dummy_probs = np.array(
        [0.99] * (len(test_df) // 2) + [0.01] * (len(test_df) - len(test_df) // 2)
    )

    # Ensure dummy_probs matches test_df length exactly
    if len(dummy_probs) < len(test_df):
        dummy_probs = np.pad(
            dummy_probs, (0, len(test_df) - len(dummy_probs)), "constant"
        )
    elif len(dummy_probs) > len(test_df):
        dummy_probs = dummy_probs[: len(test_df)]

    aug_df = create_augmented_dataset(train_df, test_df, dummy_probs)

    # Check that rows were added
    original_len = len(train_df)
    new_len = len(aug_df)
    print(f"    Original Train Size: {original_len}")
    print(f"    Augmented Train Size: {new_len}")

    assert new_len > original_len, "Augmentation failed to add rows"
    assert "Insult" in aug_df.columns, "Augmented dataset missing target column"
    print("    Augmentation logic verified.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
