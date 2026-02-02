import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_processing import load_data, StructuralFeatureExtractor, InsultDataset
from library.model import HybridDebertaModel
from library.train import train_fn, eval_fn


def main():
    print("=== Starting Library Demonstration & Verification ===")

    # ---------------------------------------------------------
    # 1. Configuration Patching (Optimize for Speed)
    # ---------------------------------------------------------
    print("\n[1] Patching Configuration for fast demonstration...")
    # Reduce dimensionality of structural features to speed up SVD
    Config.SVD_COMPONENTS = 8
    Config.TFIDF_MAX_FEATURES = 200
    # Reduce batch size for the demo
    Config.BATCH_SIZE = 4
    # Use a temporary working directory for this demo
    demo_work_dir = "./working/demo_run"
    os.makedirs(demo_work_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[2] Verifying Utilities...")
    set_seed(42)
    # Verify reproducibility
    rand_1 = np.random.rand()
    set_seed(42)
    rand_2 = np.random.rand()
    assert rand_1 == rand_2, "set_seed failed to ensure NumPy reproducibility."
    print("    -> Reproducibility check passed.")

    # ---------------------------------------------------------
    # 3. Verify Data Processing
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Processing Pipeline...")

    # A. Test Feature Extractor in isolation
    dummy_texts = [
        "You are an idiot.",
        "Have a nice day.",
        "Absolute garbage.",
        "Hello world.",
    ]
    extractor = StructuralFeatureExtractor(
        ngram_range=(2, 3), max_features=50, n_components=2
    )
    extractor.fit(dummy_texts)
    dummy_feats = extractor.transform(dummy_texts)

    assert dummy_feats.shape == (
        4,
        2,
    ), f"Extractor output shape mismatch. Expected (4, 2), got {dummy_feats.shape}"
    print("    -> StructuralFeatureExtractor logic verified.")

    # B. Load full data pipeline (using patched Config)
    # We pass load_cached_data=False to force the pipeline to run feature extraction
    print("    -> Loading data and generating features (this may take a moment)...")
    train_ds, val_ds, test_ds = load_data(load_cached_data=False)

    # C. Verify Dataset Item Structure
    sample_idx = 0
    sample = train_ds[sample_idx]

    # Check keys
    required_keys = {"input_ids", "attention_mask", "structural_features", "label"}
    assert required_keys.issubset(
        sample.keys()
    ), f"Dataset item missing keys. Found: {sample.keys()}"

    # Check Tensor Shapes
    # Input IDs should match Config.MAX_LENGTH
    assert (
        sample["input_ids"].shape[0] == Config.MAX_LENGTH
    ), "Tokenized sequence length mismatch."
    # Structural features should match our patched SVD_COMPONENTS (8)
    assert (
        sample["structural_features"].shape[0] == Config.SVD_COMPONENTS
    ), "Structural feature dimension mismatch."
    # Label should be a scalar
    assert sample["label"].numel() == 1, "Label shape mismatch."

    print(f"    -> Dataset integrity verified. Train size: {len(train_ds)}")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    device = Config.DEVICE
    print(f"    -> Using device: {device}")

    model = HybridDebertaModel(
        model_name=Config.MODEL_NAME,
        num_structural_features=Config.SVD_COMPONENTS,
        hidden_size=Config.HIDDEN_SIZE,
    )
    model.to(device)

    # Create a dummy batch to test forward pass
    # We use a DataLoader to handle collation automatically
    dummy_loader = DataLoader(Subset(train_ds, [0, 1]), batch_size=2)
    batch = next(iter(dummy_loader))

    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    struct = batch["structural_features"].to(device)

    with torch.no_grad():
        logits = model(input_ids, mask, struct)

    assert logits.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {logits.shape}"
    print("    -> Model forward pass verified.")

    # ---------------------------------------------------------
    # 5. Verify Training & Evaluation Loop
    # ---------------------------------------------------------
    print("\n[5] Verifying Training & Evaluation Steps...")

    # Create tiny subsets to simulate a quick epoch
    # 8 samples for train (2 batches of 4), 4 samples for val (1 batch of 4)
    train_subset = Subset(train_ds, range(8))
    val_subset = Subset(val_ds, range(4))

    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE)

    # Setup Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Run Training Step
    print("    -> Running train_fn on subset...")
    avg_loss = train_fn(model, train_loader, optimizer, scheduler, device, criterion)
    print(f"       Train Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss returned NaN."

    # Run Evaluation Step
    print("    -> Running eval_fn on subset...")
    val_loss, val_auc = eval_fn(model, val_loader, device, criterion)
    print(f"       Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert isinstance(val_auc, float), "AUC score is not a float."

    print("    -> Training and Evaluation loops verified.")

    # ---------------------------------------------------------
    # 6. Verify Checkpointing
    # ---------------------------------------------------------
    print("\n[6] Verifying Checkpointing...")
    ckpt_path = os.path.join(demo_work_dir, "test_model.bin")

    # Save
    save_checkpoint(model, optimizer, epoch=1, metric_value=val_auc, filename=ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    # Load
    checkpoint = load_checkpoint(ckpt_path, model, optimizer, device)
    assert checkpoint["epoch"] == 1, "Checkpoint epoch mismatch."
    assert checkpoint["metric_value"] == val_auc, "Checkpoint metric value mismatch."

    print("    -> Checkpoint save/load verified.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
