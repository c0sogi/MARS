import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Import library components
from library.utils import seed_everything, compute_auc, get_optimizer_params
from library.dataset import get_dataloaders, process_f27, ManufacturingDataset
from library.model import PostNormConformerSwiGLU
from library.trainer import Trainer


def test_utils():
    print("\n=== Testing Library Utils ===")

    # 1. Test seed_everything
    seed_everything(42)
    r1 = np.random.rand(5)
    seed_everything(42)
    r2 = np.random.rand(5)
    assert np.allclose(
        r1, r2
    ), "seed_everything failed to produce deterministic numpy results"
    print("seed_everything: Verified.")

    # 2. Test compute_auc
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    # Expected: 0.75 (Order: 0.1(0), 0.35(1-Error), 0.4(0), 0.8(1)) -> 3 correct pairs out of 4
    auc = compute_auc(y_true, y_pred)
    assert 0.0 <= auc <= 1.0, "AUC score out of range"
    print(f"compute_auc (numpy): Verified (Score: {auc:.4f})")

    # Test with tensors
    y_true_t = torch.tensor(y_true)
    y_pred_t = torch.tensor(y_pred)
    auc_t = compute_auc(y_true_t, y_pred_t)
    assert np.isclose(
        auc, auc_t
    ), "compute_auc mismatch between numpy and tensor inputs"
    print("compute_auc (tensor): Verified.")

    # 3. Test get_optimizer_params
    model = nn.Linear(10, 1)
    # Linear has weight (decay) and bias (no decay)
    params = get_optimizer_params(model, weight_decay=0.1)
    assert len(params) == 2, "Optimizer params should have 2 groups"

    # Check groups
    decay_group = params[0] if params[0]["weight_decay"] > 0 else params[1]
    no_decay_group = params[1] if params[0]["weight_decay"] > 0 else params[0]

    assert (
        decay_group["weight_decay"] == 0.1
    ), "Weight decay group has incorrect decay value"
    assert (
        no_decay_group["weight_decay"] == 0.0
    ), "No decay group has incorrect decay value"
    assert (
        len(decay_group["params"]) == 1
    ), "Expected 1 parameter in decay group (weight)"
    assert (
        len(no_decay_group["params"]) == 1
    ), "Expected 1 parameter in no_decay group (bias)"
    print("get_optimizer_params: Verified.")


def test_dataset_logic():
    print("\n=== Testing Dataset Logic ===")

    # 1. Test process_f27
    # f_27 is a string of length 10. A=0, B=1...
    dummy_series = pd.Series(["ABCDEFGHIJ", "BACDEFGHIJ"])
    processed = process_f27(dummy_series)

    expected_0 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    expected_1 = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])

    assert processed.shape == (
        2,
        10,
    ), f"Processed shape mismatch. Got {processed.shape}"
    assert np.array_equal(processed[0], expected_0), "First row encoding incorrect"
    assert np.array_equal(processed[1], expected_1), "Second row encoding incorrect"
    print("process_f27: Verified.")

    # 2. Test ManufacturingDataset directly
    cont = np.random.randn(5, 30).astype(np.float32)
    cat = np.random.randint(0, 26, (5, 10)).astype(
        np.int64
    )  # Ensure int64/long for embedding
    targets = np.random.randint(0, 2, (5,)).astype(np.float32)

    ds = ManufacturingDataset(cont, cat, targets)
    item = ds[0]

    assert "continuous" in item and "categorical" in item and "target" in item
    assert item["continuous"].shape == (30,)
    assert item["categorical"].shape == (10,)
    assert (
        item["categorical"].dtype == torch.long
    ), "Categorical features must be LongTensor"
    print("ManufacturingDataset: Verified.")


def run_integration_demo():
    print("\n=== Running Integration Demo (Subset) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Get Dataloaders (This processes data if not cached)
    # We use a large batch size for the loader creation to speed up initialization,
    # but we will subset the dataset immediately.
    print("Loading data...")
    train_loader_full, val_loader_full, test_loader_full, test_ids = get_dataloaders(
        batch_size=1024, load_cached_data=True, num_workers=0
    )

    # 2. Create Subsets for Speed
    # We only use 100 samples for training and validation to verify the pipeline quickly
    subset_size = 100
    train_subset = Subset(train_loader_full.dataset, range(subset_size))
    val_subset = Subset(val_loader_full.dataset, range(subset_size))
    test_subset = Subset(test_loader_full.dataset, range(subset_size))

    # Create new loaders for the subsets
    demo_train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    demo_val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    demo_test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)

    print(f"Created subset loaders. Train size: {len(train_subset)}")

    # 3. Instantiate Model
    model = PostNormConformerSwiGLU().to(device)
    print("Model instantiated.")

    # Verify Forward Pass
    batch = next(iter(demo_train_loader))
    cont = batch["continuous"].to(device)
    cat = batch["categorical"].to(device)

    with torch.no_grad():
        out = model(cont, cat)

    assert out.shape == (
        cont.shape[0],
    ), f"Model output shape mismatch. Expected ({cont.shape[0]},), got {out.shape}"
    assert (out >= 0).all() and (
        out <= 1
    ).all(), "Model output not in [0, 1] range (Sigmoid check)"
    print("Model forward pass: Verified.")

    # 4. Setup Trainer
    optimizer_params = get_optimizer_params(model, weight_decay=1e-2)
    optimizer = optim.AdamW(optimizer_params, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = nn.BCELoss()

    # Use a temporary directory for checkpoints
    checkpoint_dir = "./working/demo_checkpoints"

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        checkpoint_dir=checkpoint_dir,
    )

    # 5. Run Training Loop
    print("Starting training loop (2 epochs)...")
    best_auc = trainer.fit(demo_train_loader, demo_val_loader, epochs=2, patience=2)
    print(f"Training finished. Best AUC on subset: {best_auc:.4f}")

    assert os.path.exists(
        os.path.join(checkpoint_dir, "best_model.pth")
    ), "Best model checkpoint not saved."

    # 6. Run Inference
    print("Running inference on test subset...")
    preds = trainer.predict(demo_test_loader)

    assert (
        len(preds) == subset_size
    ), f"Prediction count mismatch. Expected {subset_size}, got {len(preds)}"
    print("Inference: Verified.")

    # 7. Generate Dummy Submission
    sub_df = pd.DataFrame({"id": test_ids[:subset_size], "target": preds})

    # Check format
    assert sub_df.shape == (subset_size, 2)
    assert "id" in sub_df.columns and "target" in sub_df.columns
    print("Submission generation: Verified.")

    # Cleanup
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    print("Cleanup complete.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        test_utils()
        test_dataset_logic()
        run_integration_demo()
        print("\nAll demonstrations completed successfully.")
    except AssertionError as e:
        print(f"\nValidation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
