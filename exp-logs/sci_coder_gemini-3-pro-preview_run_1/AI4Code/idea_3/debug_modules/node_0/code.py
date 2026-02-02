import sys
import os
import torch
import pandas as pd
import numpy as np

# ==========================================
# 0. Environment Setup & Patching
# ==========================================


# Patch tqdm to suppress progress bars as per requirements
class TqdmDummy:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


# Mock the module so subsequent imports use the dummy
import tqdm

tqdm.tqdm = TqdmDummy

# Import Library Modules
# (These must be imported AFTER patching tqdm)
from library.config import DEVICE, BATCH_SIZE, MODEL_SAVE_PATH, SUBMISSION_PATH, SEED
from library.utils import set_seed, collate_fn, compute_kendall_tau
from library.dataset import NotebookDataset
from library.model import CAAN
from library.engine import Engine

# ==========================================
# 1. Main Execution
# ==========================================


def run_demo():
    print("Starting End-to-End Demo...")

    # 1. Reproducibility
    set_seed(SEED)

    # 2. Verify Metric Logic
    print("\n--- Verifying Metric Logic ---")
    # Case: 3 items. Ground Truth: [0, 1, 2]. Pred: [0, 2, 1].
    # Inversions: (2, 1) -> 1 inversion.
    # Pairs: 3*(2)/2 = 3.
    # Kendall Tau = 1 - 4 * (S / (n*(n-1)))
    # n*(n-1) = 6. S=1. 1 - 4*(1/6) = 0.333...
    gt = [["a", "b", "c"]]
    pred = [["a", "c", "b"]]
    score = compute_kendall_tau(pred, gt)
    expected = 1.0 - 4.0 * (1.0 / 6.0)
    assert (
        abs(score - expected) < 1e-6
    ), f"Metric check failed. Got {score}, expected {expected}"
    print("Metric logic verified.")

    # 3. Data Pipeline (Debug Mode)
    print("\n--- Initializing Data Pipeline (Debug Mode) ---")
    # We use debug=True to process only 100 notebooks for speed.
    # We set load_cached_data=False to ensure we generate the small debug file
    # and don't accidentally load a full dataset if it exists.

    print("Processing Train Data...")
    train_dataset = NotebookDataset(split="train", load_cached_data=False, debug=True)

    print("Processing Validation Data...")
    val_dataset = NotebookDataset(split="val", load_cached_data=False, debug=True)

    # Verify dataset is not empty
    assert len(train_dataset) > 0, "Train dataset is empty."
    assert len(val_dataset) > 0, "Validation dataset is empty."

    # Create DataLoaders
    # Using a small batch size for the demo
    demo_batch_size = 8
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=demo_batch_size,
        shuffle=True,
        num_workers=0,  # 0 workers for simple sequential debugging
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=demo_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(
        f"DataLoaders ready. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    # 4. Model Initialization & Forward Check
    print("\n--- Initializing Model ---")
    model = CAAN().to(DEVICE)

    print("Verifying Forward Pass...")
    batch = next(iter(train_loader))
    code_emb = batch["code_emb"].to(DEVICE)
    code_mask = batch["code_mask"].to(DEVICE)
    md_emb = batch["md_emb"].to(DEVICE)
    md_mask = batch["md_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(code_emb, code_mask, md_emb, md_mask)

    # Check output shape: (Batch, Max_MD_Len, Max_Code_Len + 1)
    B, L_md, _ = logits.shape
    assert B == code_emb.size(0)
    assert L_md == md_emb.size(1)
    assert logits.size(2) == code_emb.size(1) + 1
    print(f"Forward pass successful. Logits shape: {logits.shape}")

    # 5. Training Loop
    print("\n--- Starting Training Demo (1 Epoch) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    engine = Engine(model, DEVICE, optimizer)

    train_loss = engine.train_one_epoch(train_loader, epoch=1)
    print(f"Train Loss: {train_loss:.4f}")

    # 6. Validation
    print("\n--- Running Validation ---")
    val_loss, val_kt = engine.validate(val_loader, val_dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Kendall Tau: {val_kt:.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 7. Inference
    print("\n--- Running Inference Demo ---")
    print("Processing Test Data...")
    test_dataset = NotebookDataset(split="test", load_cached_data=False, debug=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=demo_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    engine.generate_submission(test_loader, test_dataset)

    # Verify Submission File
    if os.path.exists(SUBMISSION_PATH):
        df_sub = pd.read_csv(SUBMISSION_PATH)
        print(f"Submission file created at {SUBMISSION_PATH}")
        print(f"Number of predictions: {len(df_sub)}")
        print("Sample prediction:")
        print(df_sub.head(1))

        # Basic format check
        assert "id" in df_sub.columns
        assert "cell_order" in df_sub.columns
        assert len(df_sub) == len(test_dataset)
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
