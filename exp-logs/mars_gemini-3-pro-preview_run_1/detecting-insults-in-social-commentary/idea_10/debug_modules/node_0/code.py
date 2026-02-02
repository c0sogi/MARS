import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, AWP
from library.data_processing import (
    load_data,
    get_structural_features,
    InsultDataset,
    prepare_student_data,
)
from library.model import HybridDeberta
from library.trainer import train_fn, valid_fn, run_fold


def main():
    # Set seed for reproducibility
    seed_everything(42)

    print(">>> 1. Configuring Environment for Fast Demonstration...")
    # Override Config attributes to run a fast, lightweight demo
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    Config.SVD_COMPONENTS = 16  # Reduced dimensions for small sample size
    Config.DEBUG_SAMPLE_SIZE = 200  # Small subset, but enough for Tfidf min_df=2
    Config.AWP_START_EPOCH = 0  # Enable AWP immediately for testing

    # Ensure working directory exists (Config creates it, but we double check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("\n>>> 2. Testing Data Loading and Processing...")
    # Load data subset
    train_df, val_df, test_df = load_data(debug=True)

    # Verify loaded sizes
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE

    # Prepare text lists
    train_texts = train_df["Comment"].fillna("").tolist()
    val_texts = val_df["Comment"].fillna("").tolist()
    test_texts = test_df["Comment"].fillna("").tolist()

    # Generate SVD Features
    # We set load_cached_data=False to force the computation logic to run
    print("    Computing SVD features...")
    svd_train, svd_val, svd_test = get_structural_features(
        train_texts, val_texts, test_texts, load_cached_data=False, debug=True
    )

    # Verify SVD feature shapes
    assert svd_train.shape == (len(train_df), Config.SVD_COMPONENTS)
    assert svd_val.shape == (len(val_df), Config.SVD_COMPONENTS)
    assert svd_test.shape == (len(test_df), Config.SVD_COMPONENTS)
    print("    SVD features computed and verified.")

    # Test Student Data Preparation (for semi-supervised logic)
    print("    Testing Student Data Prep...")
    dummy_soft_labels = np.random.rand(len(test_df)).astype(np.float32)
    s_texts, s_svd, s_labels = prepare_student_data(
        train_df, test_df, dummy_soft_labels, svd_train, svd_test
    )
    assert len(s_texts) == len(train_df) + len(test_df)
    assert len(s_labels) == len(train_df) + len(test_df)
    assert s_svd.shape[0] == len(train_df) + len(test_df)

    print("\n>>> 3. Testing Dataset and DataLoader...")
    # Create Datasets with reduced max_len for speed
    train_dataset = InsultDataset(
        texts=train_texts,
        svd_features=svd_train,
        labels=train_df["Insult"].values,
        max_len=32,
    )
    val_dataset = InsultDataset(
        texts=val_texts,
        svd_features=svd_val,
        labels=val_df["Insult"].values,
        max_len=32,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    svd_feats = batch["svd_features"]
    labels = batch["label"]

    assert input_ids.shape == (Config.TRAIN_BATCH_SIZE, 32)
    assert svd_feats.shape == (Config.TRAIN_BATCH_SIZE, Config.SVD_COMPONENTS)
    assert labels.shape == (Config.TRAIN_BATCH_SIZE,)
    print("    DataLoader verified.")

    print("\n>>> 4. Testing Model Architecture...")
    device = Config.DEVICE
    # Initialize model (pretrained=True will download weights)
    model = HybridDeberta(pretrained=True)
    model.to(device)

    # Perform dummy forward pass
    with torch.no_grad():
        logits = model(
            input_ids.to(device), attention_mask.to(device), svd_feats.to(device)
        )

    # Verify output shape: (Batch_Size, 1)
    assert logits.shape == (Config.TRAIN_BATCH_SIZE, 1)
    print("    Model forward pass successful.")

    print("\n>>> 5. Testing Training Components (Optimizer, AWP)...")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize AWP
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-2, start_epoch=0)

    # Manual Training Step
    model.train()

    # 1. Forward
    preds = model(input_ids.to(device), attention_mask.to(device), svd_feats.to(device))
    loss = criterion(preds.view(-1), labels.to(device).float())

    # 2. Backward
    loss.backward()

    # 3. AWP Attack & Restore
    awp.attack()
    # (In a real loop, we would re-forward and re-backward here)
    awp.restore()

    # 4. Optimizer Step
    optimizer.step()
    optimizer.zero_grad()

    print(f"    Manual training step complete. Loss: {loss.item():.4f}")

    print("\n>>> 6. Running Integration Test (run_fold)...")
    # Define a temporary path for the model checkpoint
    save_path = os.path.join(Config.WORKING_DIR, "demo_model.bin")

    # Execute a full fold run (1 epoch, small data)
    best_score = run_fold(
        fold=0,
        train_loader=train_loader,
        valid_loader=val_loader,
        device=device,
        save_path=save_path,
    )

    print(f"    Integration test complete. Best AUC: {best_score:.4f}")

    # Verify model file was created
    if not os.path.exists(save_path):
        raise AssertionError("Model checkpoint was not saved.")

    print("\n>>> SUCCESS: All components verified.")


if __name__ == "__main__":
    main()
