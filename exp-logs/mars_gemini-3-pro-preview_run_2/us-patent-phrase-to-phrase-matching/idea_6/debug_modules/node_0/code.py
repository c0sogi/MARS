import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything
from library.dataset import PhraseDataset
from library.model import HybridDeberta
from library.engine import train_fn, valid_fn
from library.awp import AWP
from library.feature_engineering import get_features_batch


def run_demo():
    print("=== Phrase Similarity Task: Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")
    seed_everything(42)

    # Override CFG settings for a fast demo run
    CFG.debug = True
    CFG.working_dir = "./working/demo_run/"
    CFG.output_dir = os.path.join(CFG.working_dir, "models")
    CFG.submission_dir = os.path.join(CFG.working_dir, "submission")

    # Use a tiny model to ensure the demo runs quickly without massive downloads/computations
    CFG.model_name = "prajjwal1/bert-tiny"
    CFG.target_size = 5

    # Training hyperparameters for demo
    CFG.epochs = 1
    CFG.train_batch_size = 4
    CFG.valid_batch_size = 4
    CFG.gradient_accumulation_steps = 1
    CFG.awp_start_epoch = 0  # Enable AWP immediately to verify it works
    CFG.print_freq = 5
    CFG.n_fold = 2

    # Setup directories
    if os.path.exists(CFG.working_dir):
        shutil.rmtree(CFG.working_dir)
    CFG.setup()

    print(f"Model: {CFG.model_name}")
    print(f"Device: {CFG.device}")

    # ---------------------------------------------------------
    # 2. Data Preparation (Subsetting)
    # ---------------------------------------------------------
    print("\n[2] Loading and subsetting data...")
    train_full = pd.read_csv(CFG.train_file)
    val_full = pd.read_csv(CFG.val_file)
    test_full = pd.read_csv(CFG.test_file)

    # Take a tiny subset for demonstration
    train_subset = train_full.head(20).reset_index(drop=True)
    val_subset = val_full.head(10).reset_index(drop=True)
    test_subset = test_full.head(10).reset_index(drop=True)

    print(f"Train subset: {train_subset.shape}")
    print(f"Val subset:   {val_subset.shape}")
    print(f"Test subset:  {test_subset.shape}")

    # ---------------------------------------------------------
    # 3. Feature Engineering Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Structural Feature Engineering...")
    # Test with explicit strings
    anchors = ["cat", "cat"]
    targets = ["cat food", "dog"]

    # Compute features
    feats = get_features_batch(
        anchors, targets, cache_name="demo_verify", load_cached_data=False
    )

    print(f"Features shape: {feats.shape}")
    print(f"Features sample: {feats[0]}")

    # Assertions
    assert feats.shape == (2, 3), "Expected shape (2, 3) for structural features"
    assert feats.dtype == np.float32, "Expected float32 dtype"
    # Check logic: cat vs cat food -> Jaccard > 0
    assert feats[0][1] > 0.0, "Jaccard similarity should be > 0 for 'cat' vs 'cat food'"

    # ---------------------------------------------------------
    # 4. Dataset & DataLoader Initialization
    # ---------------------------------------------------------
    print("\n[4] Initializing Datasets and DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # Create Datasets
    train_ds = PhraseDataset(
        train_subset, tokenizer, mode="train", cache_name="train_demo"
    )
    val_ds = PhraseDataset(val_subset, tokenizer, mode="val", cache_name="val_demo")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        val_ds,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Verify a single batch
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "structural_features" in batch
    assert "label" in batch
    assert batch["structural_features"].shape[1] == 3
    print("Dataset verification passed.")

    # ---------------------------------------------------------
    # 5. Model Initialization
    # ---------------------------------------------------------
    print("\n[5] Initializing HybridDeberta Model...")
    model = HybridDeberta(pretrained=True)
    model.to(CFG.device)

    # Verify Forward Pass
    with torch.no_grad():
        ids = batch["input_ids"].to(CFG.device)
        mask = batch["attention_mask"].to(CFG.device)
        struct = batch["structural_features"].to(CFG.device)
        output = model(ids, mask, struct)

    print(f"Output logits shape: {output.shape}")
    assert output.shape == (
        CFG.train_batch_size,
        CFG.target_size,
    ), "Output shape mismatch"

    # ---------------------------------------------------------
    # 6. Training Loop (with AWP)
    # ---------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")
    optimizer = AdamW(
        model.parameters(), lr=CFG.encoder_lr, weight_decay=CFG.weight_decay
    )

    num_train_steps = len(train_loader) * CFG.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Initialize Adversarial Weight Perturbation
    awp = AWP(
        model,
        optimizer,
        adv_lr=CFG.awp_lr,
        adv_eps=CFG.awp_eps,
        start_epoch=CFG.awp_start_epoch,
    )

    # Run training function
    avg_loss = train_fn(
        train_loader, model, optimizer, 0, scheduler, CFG.device, awp=awp
    )
    print(f"Training completed. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss resulted in NaN"

    # ---------------------------------------------------------
    # 7. Validation Loop
    # ---------------------------------------------------------
    print("\n[7] Running Validation Loop...")
    score, val_loss, val_preds = valid_fn(valid_loader, model, CFG.device)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Pearson Score: {score:.4f}")

    assert len(val_preds) == len(
        val_subset
    ), "Number of predictions does not match validation set size"
    assert (
        val_preds.min() >= 0.0 and val_preds.max() <= 1.0
    ), "Predictions outside valid range [0, 1]"

    # ---------------------------------------------------------
    # 8. Inference on Test Set
    # ---------------------------------------------------------
    print("\n[8] Running Inference on Test Set...")
    test_ds = PhraseDataset(test_subset, tokenizer, mode="test", cache_name="test_demo")
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    test_preds = []
    score_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to(CFG.device)

    with torch.no_grad():
        for inputs in test_loader:
            input_ids = inputs["input_ids"].to(CFG.device)
            attention_mask = inputs["attention_mask"].to(CFG.device)
            structural_features = inputs["structural_features"].to(CFG.device)

            # Forward pass
            logits = model(input_ids, attention_mask, structural_features)

            # Calculate Expected Value (0-1 range)
            probs = torch.softmax(logits, dim=1)
            expected_scores = torch.sum(probs * score_values, dim=1)

            test_preds.append(expected_scores.cpu().numpy())

    test_predictions = np.concatenate(test_preds)

    # ---------------------------------------------------------
    # 9. Submission Generation
    # ---------------------------------------------------------
    print("\n[9] Generating Submission File...")
    submission = pd.DataFrame({"id": test_subset["id"], "score": test_predictions})

    print(submission.head())

    os.makedirs(CFG.submission_dir, exist_ok=True)
    submission_path = os.path.join(CFG.submission_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file was not created"
    print(f"Submission saved to: {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
