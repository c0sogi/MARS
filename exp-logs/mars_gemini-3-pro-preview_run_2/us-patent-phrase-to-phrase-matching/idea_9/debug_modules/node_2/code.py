import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from transformers import AutoTokenizer

# Ensure the library modules can be imported
sys.path.append("./library")

# Import from the provided files
from library.config import CFG
from library.utils import seed_everything, get_score, AverageMeter
from library.data import (
    get_levenshtein_distance,
    get_cpc_texts,
    get_structural_features,
    get_data_loaders,
    get_test_loader,
    process_data,
)
from library.model import CustomDeberta
from library.engine import train_fn, valid_fn
from library.stacking import train_stacking_model


def run_demo():
    print("=== Starting Demo Run ===")

    # 1. Configuration Overrides for Speed and Demo
    print("\n[1] Configuring environment...")
    CFG.debug = True
    CFG.epochs = 1
    CFG.model_name = "prajjwal1/bert-tiny"  # Use a tiny model for rapid execution
    CFG.output_dir = "./working/demo_run"
    CFG.train_processed_path = os.path.join(CFG.output_dir, "train_processed.parquet")
    CFG.val_processed_path = os.path.join(CFG.output_dir, "val_processed.parquet")
    CFG.test_processed_path = os.path.join(CFG.output_dir, "test_processed.parquet")
    CFG.submission_path = os.path.join(CFG.output_dir, "submission/submission.csv")
    CFG.batch_size = 4
    CFG.n_fold = 2  # Reduce folds for stacking demo

    # Clean up demo directory if exists
    if os.path.exists(CFG.output_dir):
        shutil.rmtree(CFG.output_dir)
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(CFG.submission_path), exist_ok=True)

    seed_everything(CFG.seed)
    device = CFG.device
    print(f"Device: {device}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test Levenshtein
    dist = get_levenshtein_distance("kitten", "sitting")
    assert dist == 3, f"Levenshtein distance incorrect. Expected 3, got {dist}"
    print("  - Levenshtein distance: OK")

    # Test Pearson Score
    y_true = [0, 0.5, 1.0]
    y_pred = [0, 0.5, 1.0]
    score = get_score(y_true, y_pred)
    assert (
        abs(score - 1.0) < 1e-6
    ), f"Pearson score incorrect. Expected 1.0, got {score}"
    print("  - Pearson score: OK")

    # 3. Data Pipeline Verification
    print("\n[3] Verifying Data Pipeline...")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # Get Loaders (Debug mode will load small subset)
    train_loader, val_loader = get_data_loaders(tokenizer, load_cached_data=False)
    test_loader, test_df = get_test_loader(tokenizer, load_cached_data=False)

    print(f"  - Train Loader length: {len(train_loader)}")
    print(f"  - Val Loader length: {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    struct_feats = batch["structural_features"]
    labels = batch["label"]

    assert input_ids.shape == (
        CFG.batch_size,
        CFG.max_len,
    ), f"Input shape mismatch. Expected {(CFG.batch_size, CFG.max_len)}, got {input_ids.shape}"
    assert struct_feats.shape == (
        CFG.batch_size,
        3,
    ), f"Structural features shape mismatch. Expected {(CFG.batch_size, 3)}, got {struct_feats.shape}"
    assert labels.shape == (
        CFG.batch_size,
    ), f"Labels shape mismatch. Expected {(CFG.batch_size,)}, got {labels.shape}"

    print("  - Batch structure and shapes: OK")

    # 4. Model Instantiation and Forward Pass
    print("\n[4] Verifying Model Architecture...")
    model = CustomDeberta(pretrained_model_name=CFG.model_name)
    model.to(device)

    # Forward pass with the batch fetched earlier
    with torch.no_grad():
        # Ensure inputs are on device
        ids = input_ids.to(device)
        mask = batch["attention_mask"].to(device)
        feats = struct_feats.to(device)

        logits = model(ids, mask, feats)

    assert logits.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), f"Model output shape mismatch. Expected {(CFG.batch_size, CFG.num_classes)}, got {logits.shape}"
    print("  - Model forward pass: OK")

    # 5. Training Engine Verification
    print("\n[5] Verifying Training Engine...")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run one training epoch
    print("  - Running training epoch (debug subset)...")
    avg_loss = train_fn(
        fold=0,
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=None,
        device=device,
    )
    print(f"    Train Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # Run validation
    print("  - Running validation...")
    val_loss, val_score, val_probs = valid_fn(val_loader, model, criterion, device)
    print(f"    Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}")
    assert (
        val_probs.shape[1] == CFG.num_classes
    ), "Validation probabilities shape mismatch"
    print("  - Engine execution: OK")

    # 6. Stacking Verification
    print("\n[6] Verifying Stacking Module...")

    # Create dummy data for stacking to run quickly without relying on full dataset processing
    # We simulate the output of the 5-fold CV (or 2-fold in this demo config)
    N_train = 50
    N_test = 20

    # Create dummy train/test dataframes
    dummy_train = pd.DataFrame(
        {
            "id": [f"train_{i}" for i in range(N_train)],
            "anchor": ["anchor"] * N_train,
            "target": ["target"] * N_train,
            "context": ["A47"] * N_train,
            "score": np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0], N_train),
        }
    )

    dummy_test = pd.DataFrame(
        {
            "id": [f"test_{i}" for i in range(N_test)],
            "anchor": ["anchor"] * N_test,
            "target": ["target"] * N_test,
            "context": ["G06"] * N_test,
        }
    )

    # Create dummy probability predictions (Level 1 output)
    # Shape: (N_samples, 5 classes)
    dummy_oof_probs = np.random.rand(N_train, 5)
    dummy_oof_probs = dummy_oof_probs / dummy_oof_probs.sum(axis=1, keepdims=True)

    dummy_test_probs = np.random.rand(N_test, 5)
    dummy_test_probs = dummy_test_probs / dummy_test_probs.sum(axis=1, keepdims=True)

    # Run stacking training
    # Note: We pass load_cached_data=False to force re-computation with our dummy data
    print("  - Training LightGBM stacking model...")
    final_preds = train_stacking_model(
        dummy_train,
        dummy_test,
        dummy_oof_probs,
        dummy_test_probs,
        load_cached_data=False,
    )

    assert len(final_preds) == N_test, "Stacking predictions length mismatch"
    assert os.path.exists(CFG.submission_path), "Submission file was not created"

    # Verify submission file content
    sub_df = pd.read_csv(CFG.submission_path)
    assert sub_df.shape == (N_test, 2), f"Submission shape mismatch: {sub_df.shape}"
    assert (
        "id" in sub_df.columns and "score" in sub_df.columns
    ), "Submission columns mismatch"
    print("  - Stacking and Submission: OK")

    print("\n=== Demo Run Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
