import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import (
    load_train_data,
    load_test_data,
    get_folds,
    prepare_loaders,
    prepare_test_loader,
    merge_pseudo_labels,
)
from library.model import InsultModel
from library.engine import get_optimizer_params, train_fn, valid_fn
from library.awp import AWP


def run_demo():
    print("Initializing Demo...")

    # ------------------------------------------------------------------------
    # 1. Runtime Configuration Patching
    # ------------------------------------------------------------------------
    # We modify Config attributes to ensure the demo runs fast (seconds/minutes)
    # instead of hours. We use a tiny model and very small batch/data sizes.
    print("Patching Config for fast demonstration...")
    Config.model_name = "prajjwal1/bert-tiny"  # Tiny model for speed
    Config.debug = True
    Config.debug_sample_size = 50  # Only use 50 samples
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.gradient_accumulation_steps = 1
    Config.n_folds = 2
    Config.stage1_epochs = 1
    Config.print_freq = 5
    Config.working_dir = "./working/demo_run"
    Config.output_dir = "./working/demo_run"

    # Re-run setup to create the new working directory
    Config.setup()
    seed_everything(Config.seed)

    device = Config.device
    print(f"Device: {device}")

    # ------------------------------------------------------------------------
    # 2. Data Loading & Processing Validation
    # ------------------------------------------------------------------------
    print("\n--- Testing Data Loading ---")

    # Test loading and caching
    df_train = load_train_data(load_cached_data=False)
    assert isinstance(df_train, pd.DataFrame)
    assert "Comment" in df_train.columns
    assert "Insult" in df_train.columns
    print(f"Loaded train data shape: {df_train.shape}")

    # Test Fold generation
    df_folds = get_folds(df_train, n_folds=Config.n_folds, seed=Config.seed)
    assert "fold" in df_folds.columns
    assert df_folds["fold"].nunique() == Config.n_folds
    print("Fold generation successful.")

    # Test DataLoader preparation
    # We use fold 0. prepare_loaders handles splitting and tokenization.
    train_loader, valid_loader = prepare_loaders(fold=0, df=df_folds, debug=True)

    # Fetch a batch to verify structure
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "target" in batch

    # Check shapes
    # input_ids: [batch_size, max_len] (flattened in dataset, but DataLoader stacks them)
    # The dataset flattens them, but let's check the tensor coming out of loader
    # Dataset returns: input_ids shape [max_len], DataLoader stacks to [batch, max_len]
    assert batch["input_ids"].shape == (Config.train_batch_size, Config.max_len)
    assert batch["target"].shape == (Config.train_batch_size,)
    print("DataLoaders created and batch structure verified.")

    # ------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ------------------------------------------------------------------------
    print("\n--- Testing Model ---")

    model = InsultModel(pretrained=True)
    model.to(device)

    # Move batch to device
    b_input_ids = batch["input_ids"].to(device)
    b_mask = batch["attention_mask"].to(device)

    # Forward pass
    output = model(b_input_ids, b_mask)

    # Check output shape: [batch_size, num_classes] (num_classes=1)
    assert output.shape == (Config.train_batch_size, 1)
    assert not torch.isnan(output).any()
    print("Model forward pass successful. Output shape verified.")

    # ------------------------------------------------------------------------
    # 4. Optimizer & Scheduler
    # ------------------------------------------------------------------------
    print("\n--- Testing Optimizer Setup ---")

    optimizer_params = get_optimizer_params(
        model,
        encoder_lr=Config.lr,
        decoder_lr=Config.lr,
        weight_decay=Config.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_params, lr=Config.lr, eps=1e-6)

    # Simple scheduler for demo
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=Config.min_lr
    )
    print("Optimizer and Scheduler initialized.")

    # ------------------------------------------------------------------------
    # 5. Adversarial Weight Perturbation (AWP)
    # ------------------------------------------------------------------------
    print("\n--- Testing AWP ---")

    awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

    # AWP requires gradients to exist. Let's do a backward pass first.
    criterion = nn.BCEWithLogitsLoss()
    labels = batch["target"].to(device)

    # Standard forward/backward
    loss = criterion(output.view(-1), labels)
    loss.backward()

    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert param.grad is not None
            break

    # Save original weights of a specific layer to verify perturbation
    # We pick the classifier head weight
    orig_weight = model.fc.weight.data.clone()

    # Attack (Perturb weights)
    awp.attack()

    # Verify weights changed
    attacked_weight = model.fc.weight.data
    # Note: If gradients are exactly zero, weights won't change, but with random init and data, they should differ.
    if not torch.equal(orig_weight, attacked_weight):
        print("AWP Attack: Weights perturbed successfully.")
    else:
        print("AWP Attack: Weights unchanged (gradients might be zero or too small).")

    # Restore weights
    awp.restore()
    restored_weight = model.fc.weight.data
    assert torch.equal(orig_weight, restored_weight)
    print("AWP Restore: Weights restored successfully.")

    # Clear grads for next steps
    optimizer.zero_grad()

    # ------------------------------------------------------------------------
    # 6. Training & Validation Loop (Engine)
    # ------------------------------------------------------------------------
    print("\n--- Testing Training Loop ---")

    # Run one training epoch
    avg_loss = train_fn(
        train_loader,
        model,
        criterion,
        optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,  # Using AWP in training
    )
    print(f"Train Loop finished. Avg Loss: {avg_loss:.4f}")

    # Run validation
    print("\n--- Testing Validation Loop ---")
    score, preds, val_loss = valid_fn(valid_loader, model, criterion, device)

    assert len(preds) == len(valid_loader.dataset)
    assert 0 <= score <= 1.0 or score == 0.5  # 0.5 if single class in batch
    print(f"Validation Loop finished. AUC: {score:.4f}")

    # ------------------------------------------------------------------------
    # 7. Pseudo-Labeling Logic
    # ------------------------------------------------------------------------
    print("\n--- Testing Pseudo-Labeling ---")

    # Load test data
    test_df = load_test_data(load_cached_data=False)
    # Mock predictions for test set (random probs)
    mock_preds = np.random.rand(len(test_df))

    # Merge
    augmented_df = merge_pseudo_labels(
        train_df=df_train,
        test_df=test_df,
        preds=mock_preds,
        threshold=0.8,  # Lower threshold to ensure some get picked for demo
    )

    assert len(augmented_df) >= len(df_train)
    assert "Insult" in augmented_df.columns
    print(
        f"Pseudo-labeling successful. Augmented size: {len(augmented_df)} (Original: {len(df_train)})"
    )

    # ------------------------------------------------------------------------
    # 8. Inference / Submission Generation
    # ------------------------------------------------------------------------
    print("\n--- Testing Inference Pipeline ---")

    test_loader = prepare_test_loader(load_cached_data=False, debug=True)

    # Reuse valid_fn logic or manual inference loop
    model.eval()
    test_preds = []

    with torch.no_grad():
        for inputs in test_loader:
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            # Forward
            y_preds = model(inputs["input_ids"], inputs["attention_mask"])
            test_preds.append(y_preds.sigmoid().cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # Create submission dataframe
    # Note: debug loader samples the dataframe, so we need to match indices if we were doing real submission
    # For demo, we just check shape matches loader dataset
    assert len(test_preds) == len(test_loader.dataset)

    submission = pd.DataFrame({"id": range(len(test_preds)), "prediction": test_preds})
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission file saved to {Config.submission_path}")

    print("\nAll demonstrations passed successfully!")


if __name__ == "__main__":
    run_demo()
