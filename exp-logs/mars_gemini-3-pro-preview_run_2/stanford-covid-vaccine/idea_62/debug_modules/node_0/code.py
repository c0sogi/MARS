import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import shutil

# Import from the provided library
from library.config import (
    SEQ_LEN,
    SCORED_LEN,
    TARGET_COLS,
    SCORED_TARGETS,
    WORKING_DIR,
    TRAIN_CSV,
)
from library.utils import parse_list_column, get_structure_adj, MCRMSELoss, MCRMSE
from library.data import process_data, RNADataset, get_features
from library.model import HS_GFN
from library.train import train_one_epoch, validate, predict, get_scored_indices


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def demo_utils():
    print("\n=== Demo: Library Utils ===")

    # 1. Test parse_list_column
    list_str = "[0.1, 0.2, 0.3]"
    parsed = parse_list_column(list_str)
    print(f"Parsed '{list_str}': {parsed}, Type: {type(parsed)}")
    assert isinstance(parsed, np.ndarray)
    assert np.allclose(parsed, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    # 2. Test get_structure_adj
    # Structure: ((..)) -> Indices: 0 pairs with 5, 1 pairs with 4. 2,3 unpaired.
    structure = "((..))"
    adj = get_structure_adj(structure)
    expected = np.array([5, 4, -1, -1, 1, 0])
    print(f"Structure '{structure}' adjacency: {adj}")
    assert np.array_equal(adj, expected)

    # 3. Test MCRMSELoss
    criterion = MCRMSELoss()
    # Create dummy preds and targets: (Batch=1, Seq=2, Channels=2)
    preds = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float32)
    targets = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
    # Errors: (0.5)^2 = 0.25. Mean = 0.25. RMSE = 0.5.
    loss = criterion(preds, targets)
    print(f"MCRMSE Loss (Expected ~0.5): {loss.item():.4f}")
    assert abs(loss.item() - 0.5) < 1e-4

    # 4. Test MCRMSE Accumulator
    metric = MCRMSE()
    metric.update(preds, targets)
    score = metric.compute()
    print(f"MCRMSE Metric Score (Expected ~0.5): {score:.4f}")
    assert abs(score - 0.5) < 1e-4
    print("Utils verification passed.")


def demo_data_pipeline():
    print("\n=== Demo: Data Pipeline ===")

    # Create a mini dataset from the metadata
    mini_csv_path = os.path.join(WORKING_DIR, "mini_train_demo.csv")
    mini_cache_path = os.path.join(WORKING_DIR, "mini_train_cache_demo.npz")

    # Read first 5 rows of metadata/train.csv
    df = pd.read_csv(TRAIN_CSV)
    df_mini = df.head(5).copy()
    df_mini.to_csv(mini_csv_path, index=False)
    print(f"Created mini dataset at {mini_csv_path} with {len(df_mini)} samples.")

    # Test get_features directly
    seq = "AGUC" * 20  # length 80
    struct = "((..))" * 10  # length 60... needs to match
    # Let's use a valid sample from the dataframe
    sample_seq = df_mini.iloc[0]["sequence"]
    sample_struct = df_mini.iloc[0]["structure"]
    sample_loop = df_mini.iloc[0]["predicted_loop_type"]

    feats, p_idx = get_features(sample_seq, sample_struct, sample_loop)
    print(f"Feature shape: {feats.shape} (Expected: 19, {len(sample_seq)})")
    assert feats.shape == (19, len(sample_seq))
    assert p_idx.shape == (len(sample_seq),)

    # Test process_data
    # Force load_cached_data=False to ensure processing logic runs
    features, partner_indices, targets, ids = process_data(
        mini_csv_path, mini_cache_path, mode="train", load_cached_data=False
    )

    print(f"Processed Batch Shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Partner Indices: {partner_indices.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  IDs: {ids.shape}")

    assert features.shape[0] == 5
    assert features.shape[1] == 19
    assert features.shape[2] == SEQ_LEN
    assert targets.shape[1] == 5  # 5 targets

    # Create Dataset and Loader
    dataset = RNADataset(features, partner_indices, targets, ids)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    print("Data pipeline verification passed.")
    return loader


def demo_model(loader):
    print("\n=== Demo: Model Architecture ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HS_GFN(in_channels=19).to(device)

    # Get a batch
    x, p_idx, y = next(iter(loader))
    x = x.to(device)
    p_idx = p_idx.to(device)
    y = y.to(device)

    print(f"Input X shape: {x.shape}")
    print(f"Input Partner Indices shape: {p_idx.shape}")

    # Pass 1: No feedback
    preds_1 = model(x, p_idx, y_prev=None)
    print(f"Output Pass 1 shape: {preds_1.shape}")
    assert preds_1.shape == (x.shape[0], SEQ_LEN, 5)

    # Pass 2: With feedback
    # Feedback expects (B, 5, L)
    y_feedback = preds_1.detach().permute(0, 2, 1)
    preds_2 = model(x, p_idx, y_prev=y_feedback)
    print(f"Output Pass 2 shape: {preds_2.shape}")
    assert preds_2.shape == (x.shape[0], SEQ_LEN, 5)

    print("Model architecture verification passed.")
    return model, device


def demo_training_functions(model, loader, device):
    print("\n=== Demo: Training Functions ===")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = MCRMSELoss()
    scored_indices = get_scored_indices()
    print(f"Scored Indices: {scored_indices}")

    # 1. Train One Epoch
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(
        model, loader, criterion, optimizer, device, scored_indices
    )
    print(f"Train Loss: {train_loss:.6f}")
    assert train_loss > 0

    # 2. Validate
    print("Running validate...")
    val_score = validate(model, loader, device, scored_indices)
    print(f"Validation Score: {val_score:.6f}")
    assert val_score >= 0

    # 3. Predict
    print("Running predict...")
    preds_arr, ids_arr = predict(model, loader, device)
    print(f"Prediction Array Shape: {preds_arr.shape}")
    print(f"IDs Array Shape: {ids_arr.shape}")

    assert preds_arr.shape[0] == len(loader.dataset)
    assert preds_arr.shape[1] == SEQ_LEN
    assert preds_arr.shape[2] == 5

    print("Training functions verification passed.")


if __name__ == "__main__":
    set_seed(42)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    try:
        # Run Demos
        demo_utils()
        loader = demo_data_pipeline()
        model, device = demo_model(loader)
        demo_training_functions(model, loader, device)

        print("\nAll demonstrations completed successfully.")

    finally:
        # Cleanup temporary files
        mini_csv = os.path.join(WORKING_DIR, "mini_train_demo.csv")
        mini_cache = os.path.join(WORKING_DIR, "mini_train_cache_demo.npz")

        if os.path.exists(mini_csv):
            os.remove(mini_csv)
        if os.path.exists(mini_cache):
            os.remove(mini_cache)
        print("Cleanup completed.")
