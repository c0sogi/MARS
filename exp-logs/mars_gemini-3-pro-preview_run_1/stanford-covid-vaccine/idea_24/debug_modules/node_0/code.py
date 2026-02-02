import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Import provided library modules
import library.config
from library.config import Config
from library.utils import seed_everything, mcrmse, format_submission
from library.dataset import RNADataset
from library.model import TokenAdaptiveWideResBiGRU
from library.engine import train_fn, eval_fn, inference_fn, masked_mse_loss


def run_demo():
    print("Starting Library Usage Demonstration...")

    # =========================================================================
    # 1. Configuration Setup
    # =========================================================================
    print("\n[1] Configuring environment for demo...")

    # Override Config attributes for a lightweight and fast run
    # We modify the class attributes directly so all imported modules see the changes
    library.config.Config.EXPERIMENT_NAME = "demo_run"
    library.config.Config.WORKING_DIR = "./working/demo_run"
    library.config.Config.CACHE_DIR = os.path.join(
        library.config.Config.WORKING_DIR, "cache"
    )
    library.config.Config.CHECKPOINT_DIR = os.path.join(
        library.config.Config.WORKING_DIR, "checkpoints"
    )
    library.config.Config.SUBMISSION_DIR = os.path.join(
        library.config.Config.WORKING_DIR, "submission"
    )
    library.config.Config.BEST_MODEL_PATH = os.path.join(
        library.config.Config.CHECKPOINT_DIR, "best_model.pth"
    )
    library.config.Config.SUBMISSION_FILE = os.path.join(
        library.config.Config.SUBMISSION_DIR, "submission.csv"
    )

    # Reduce model size for speed
    library.config.Config.HIDDEN_DIM = 64  # Reduced from 384
    library.config.Config.NUM_LAYERS = 2  # Reduced from 6
    library.config.Config.EMBED_DIM = 32
    library.config.Config.LOOP_EMBED_DIM = 16
    library.config.Config.DIST_EMBED_DIM = 16

    # Reduce training parameters
    library.config.Config.BATCH_SIZE = 4
    library.config.Config.EPOCHS = 1
    library.config.Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Config: Layers={Config.NUM_LAYERS}, Hidden={Config.HIDDEN_DIM}")

    # =========================================================================
    # 2. Utils Demonstration
    # =========================================================================
    print("\n[2] Testing Utility Functions...")

    # Test seed_everything
    seed_everything(42)
    rnd1 = np.random.rand(5)
    seed_everything(42)
    rnd2 = np.random.rand(5)
    assert np.allclose(
        rnd1, rnd2
    ), "seed_everything failed to reproduce numpy random numbers"
    print(" - seed_everything: Verified.")

    # Test mcrmse
    # Create dummy predictions and targets (Batch=10, Seq=68, Targets=3)
    y_true = np.random.rand(10, 68, 3)
    y_pred = y_true + 0.1  # Constant error
    score = mcrmse(y_true, y_pred)
    # RMSE of constant 0.1 error is 0.1. Mean of RMSEs is 0.1.
    assert np.isclose(
        score, 0.1, atol=1e-5
    ), f"mcrmse calculation incorrect. Expected ~0.1, got {score}"
    print(f" - mcrmse: Verified (Score: {score:.4f}).")

    # Test format_submission
    dummy_preds = np.zeros((2, 68, 3), dtype=np.float32)
    dummy_ids = ["id_test_001", "id_test_002"]
    dummy_sub_path = os.path.join(Config.WORKING_DIR, "dummy_submission.csv")

    format_submission(dummy_preds, dummy_ids, dummy_sub_path)
    assert os.path.exists(dummy_sub_path), "Submission file was not created."

    df_sub = pd.read_csv(dummy_sub_path)
    # Expected rows: 2 samples * 107 length = 214 rows
    assert (
        len(df_sub) == 2 * 107
    ), f"Submission length mismatch. Expected 214, got {len(df_sub)}"
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."
    print(" - format_submission: Verified.")

    # =========================================================================
    # 3. Dataset Demonstration
    # =========================================================================
    print("\n[3] Testing RNADataset...")

    # Initialize Train Dataset
    # This will trigger processing and caching
    train_dataset = RNADataset(mode="train", load_cached_data=False)
    print(f" - Train Dataset loaded. Size: {len(train_dataset)}")

    # Validate a single item
    item = train_dataset[0]
    required_keys = ["sequence", "loop_type", "pair_dist", "targets", "id"]
    for k in required_keys:
        assert k in item, f"Missing key {k} in dataset item."

    # Check shapes
    seq_len = Config.SEQ_LEN
    assert item["sequence"].shape == (
        seq_len,
    ), f"Sequence shape mismatch: {item['sequence'].shape}"
    assert item["pair_dist"].shape == (
        seq_len,
    ), f"Pair dist shape mismatch: {item['pair_dist'].shape}"
    assert item["targets"].shape == (
        seq_len,
        3,
    ), f"Targets shape mismatch: {item['targets'].shape}"

    # Check dtypes
    assert item["sequence"].dtype == torch.long, "Sequence dtype should be long"
    assert item["pair_dist"].dtype == torch.float32, "Pair dist dtype should be float32"

    print(" - Dataset item structure verified.")

    # =========================================================================
    # 4. Model Demonstration
    # =========================================================================
    print("\n[4] Testing TokenAdaptiveWideResBiGRU Model...")

    device = Config.DEVICE
    model = TokenAdaptiveWideResBiGRU().to(device)

    # Create a batch from the dataset item
    batch_seq = item["sequence"].unsqueeze(0).to(device)  # (1, 107)
    batch_loop = item["loop_type"].unsqueeze(0).to(device)  # (1, 107)
    batch_dist = item["pair_dist"].unsqueeze(0).to(device)  # (1, 107)

    # Forward pass
    with torch.no_grad():
        output = model(batch_seq, batch_loop, batch_dist)

    # Verify output
    # Expected shape: (Batch, Seq_Len, Num_Targets) -> (1, 107, 3)
    assert output.shape == (
        1,
        seq_len,
        3,
    ), f"Model output shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print(" - Model forward pass successful. Output shape verified.")

    # =========================================================================
    # 5. Engine Demonstration (Training Loop)
    # =========================================================================
    print("\n[5] Testing Engine (Train/Eval/Inference)...")

    # Create small dataloaders for speed
    subset_indices = list(range(10))  # Use only 10 samples
    train_subset = Subset(train_dataset, subset_indices)

    # We need a validation dataset as well
    val_dataset = RNADataset(mode="val", load_cached_data=False)
    val_subset = Subset(val_dataset, subset_indices)

    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Test train_fn
    print(" - Running training step...")
    train_loss = train_fn(model, train_loader, optimizer, device)
    assert isinstance(train_loss, float), "train_fn did not return a float loss"
    print(f"   Train Loss: {train_loss:.6f}")

    # Test eval_fn
    print(" - Running evaluation step...")
    val_score = eval_fn(model, val_loader, device)
    assert isinstance(val_score, float), "eval_fn did not return a float score"
    print(f"   Val Score (MCRMSE): {val_score:.6f}")

    # Test inference_fn
    print(" - Running inference step...")
    # Using val subset as proxy for test to avoid loading another large file,
    # but strictly speaking we should use test mode. Let's load test briefly.
    test_dataset = RNADataset(mode="test", load_cached_data=False)
    test_subset = Subset(test_dataset, list(range(5)))
    test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    preds, ids = inference_fn(model, test_loader, device)

    # Verify inference output
    # Preds shape should be (Num_Samples, Pred_Len, 3) -> (5, 68, 3)
    assert preds.shape == (
        5,
        Config.PRED_LEN,
        3,
    ), f"Inference shape mismatch: {preds.shape}"
    assert len(ids) == 5, "Inference ID count mismatch"
    print("   Inference successful.")

    # Save dummy model for completeness check
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint not saved."

    # =========================================================================
    # 6. Final Output Generation
    # =========================================================================
    print("\n[6] Generating final submission from inference...")
    format_submission(preds, ids, Config.SUBMISSION_FILE)

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
