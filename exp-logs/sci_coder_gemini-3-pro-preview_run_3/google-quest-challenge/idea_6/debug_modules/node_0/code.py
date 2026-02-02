import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import create_folds, get_tokenizer, QuestDataset
from library.models import SegmentAwareNet
from library.engine import train_fn, eval_fn, extract_features
from library.stacking import StackingTrainer


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides for Demo
    # --------------------------------------------------------------------------
    print("--- 1. Setup & Configuration ---")
    seed_everything(Config.SEED)

    # Override Config to use a tiny model and temporary directory for speed
    Config.WORKING_DIR = "./working/demo_run/"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Use a tiny BERT model for demonstration purposes
    DEMO_BACKBONE = "prajjwal1/bert-tiny"
    Config.BACKBONES = {
        "demo_model": {"name": DEMO_BACKBONE, "lr": 1e-4, "batch_size": 8}
    }

    Config.EPOCHS = 1
    Config.N_FOLDS = 2  # Reduce folds for demo
    Config.BATCH_SIZE = 8

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model: {DEMO_BACKBONE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # --------------------------------------------------------------------------
    print("\n--- 2. Data Loading & Preprocessing ---")

    # Generate folds (logic from library/dataset.py)
    # We force re-computation to ensure it works, then slice for speed
    full_train_df = create_folds(load_cached_data=False)

    # Slice datasets to a very small subset for the demo
    # We take 40 samples for training (20 per fold) and 10 for testing
    subset_indices = np.arange(40)
    train_df = full_train_df.iloc[subset_indices].copy()

    # Manually assign folds 0 and 1 to ensure we have data for both folds in our subset
    train_df["fold"] = np.concatenate([np.zeros(20), np.ones(20)]).astype(int)

    # Load test data and take a subset
    test_df_full = pd.read_csv(Config.TEST_PATH)
    test_df = test_df_full.iloc[:10].copy()

    print(f"Train Subset Shape: {train_df.shape}")
    print(f"Test Subset Shape: {test_df.shape}")

    # Initialize Tokenizer
    tokenizer = get_tokenizer(DEMO_BACKBONE)

    # Verify Dataset Class
    ds_check = QuestDataset(train_df.iloc[:2], tokenizer, max_len=128)
    item = ds_check[0]
    print("Dataset Item Keys:", item.keys())
    assert "input_ids" in item
    assert "q_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 128

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # --------------------------------------------------------------------------
    print("\n--- 3. Model Initialization ---")

    device = Config.DEVICE
    model = SegmentAwareNet(DEMO_BACKBONE, pretrained=True)
    model.to(device)

    # Verify forward pass with dummy batch
    print("Verifying forward pass...")
    dummy_batch = {
        k: v.unsqueeze(0).to(device) for k, v in item.items() if k != "labels"
    }
    # Add dummy labels for loss calculation
    dummy_labels = torch.rand((1, 30)).to(device)

    with torch.no_grad():
        output = model(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
            q_mask=dummy_batch["q_mask"],
            a_mask=dummy_batch["a_mask"],
            labels=dummy_labels,
        )

    assert "logits" in output
    assert "features" in output
    assert "loss" in output
    assert output["logits"].shape == (1, 30)
    # Feature dim for bert-tiny (hidden=128) should be 128 * 4 = 512
    assert output["features"].shape == (1, 128 * 4)
    print("Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 4. Training Loop & Feature Extraction (Simulating Cross-Validation)
    # --------------------------------------------------------------------------
    print("\n--- 4. Training & Feature Extraction Loop ---")

    # Containers for OOF (Out-Of-Fold) features and Test features
    # We need to collect features to pass to the StackingTrainer
    oof_features = np.zeros((len(train_df), 128 * 4))
    test_features_accum = np.zeros((Config.N_FOLDS, len(test_df), 128 * 4))

    # Create Test DataLoader once
    test_ds = QuestDataset(test_df, tokenizer, max_len=128, inference=True)
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    for fold in range(Config.N_FOLDS):
        print(f"\nProcessing Fold {fold}...")

        # Split Data
        train_idx = train_df[train_df["fold"] != fold].index
        val_idx = train_df[train_df["fold"] == fold].index

        fold_train_df = train_df.loc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.loc[val_idx].reset_index(drop=True)

        # DataLoaders
        train_ds = QuestDataset(fold_train_df, tokenizer, max_len=128)
        val_ds = QuestDataset(fold_val_df, tokenizer, max_len=128)

        train_loader = DataLoader(
            train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Re-init model for each fold
        model = SegmentAwareNet(DEMO_BACKBONE, pretrained=True)
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        # Train (1 epoch)
        loss = train_fn(train_loader, model, optimizer, device)
        print(f"  Fold {fold} Train Loss: {loss:.4f}")

        # Eval
        val_loss, val_score = eval_fn(val_loader, model, device)
        print(f"  Fold {fold} Val Loss: {val_loss:.4f} | Spearman: {val_score:.4f}")

        # Extract Features for Validation Set (OOF)
        val_feats, _ = extract_features(val_loader, model, device)

        # Map back to original indices using iloc locations relative to the subset
        # Note: In the loop, val_idx are indices of the 'train_df' subset.
        # Since 'train_df' was created with reset_index implicitly via iloc earlier,
        # we need to be careful. Here 'val_idx' are the integer indices in train_df.
        # We fill the oof_features array at these positions.
        # Since train_df indices are 0..39, this works directly.
        oof_features[val_idx] = val_feats

        # Extract Features for Test Set
        test_feats, _ = extract_features(test_loader, model, device)
        test_features_accum[fold] = test_feats

    # Average test features across folds
    avg_test_features = np.mean(test_features_accum, axis=0)

    print(f"\nOOF Features Shape: {oof_features.shape}")
    print(f"Test Features Shape: {avg_test_features.shape}")

    # --------------------------------------------------------------------------
    # 5. Stacking (Level 1 & Level 2)
    # --------------------------------------------------------------------------
    print("\n--- 5. Stacking Ensemble ---")

    stacker = StackingTrainer()

    # Prepare targets
    targets = train_df[Config.TARGET_COLS].values
    folds = train_df["fold"].values

    # 5a. Train Level 1 Ridge (Base Model Refinement)
    # We pass the features we just extracted.
    # The StackingTrainer will perform internal CV using Ridge to generate "calibrated" predictions.
    l1_results = stacker.train_l1_ridge(
        backbone_name="demo_model",
        features=oof_features,
        targets=targets,
        test_features=avg_test_features,
        folds=folds,
        load_cached=False,  # Force training
    )

    assert "oof_preds" in l1_results
    assert "test_preds" in l1_results
    assert l1_results["oof_preds"].shape == (40, 30)

    # 5b. Train Level 2 Meta-Learner
    # In a real scenario, we would have multiple backbones in this dictionary.
    l1_outputs = {"demo_model": l1_results}

    final_preds = stacker.train_l2_meta(
        l1_outputs=l1_outputs, targets=targets, load_cached=False
    )

    print(f"Final Predictions Shape: {final_preds.shape}")
    assert final_preds.shape == (10, 30)
    assert (final_preds >= 0).all() and (final_preds <= 1).all()

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- 6. Generating Submission ---")

    test_ids = test_df["qa_id"].values
    stacker.save_submission(final_preds, test_ids)

    # Verify submission file exists and format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(sub_df.head(2))

    assert sub_df.shape == (10, 31)  # qa_id + 30 targets
    assert "qa_id" in sub_df.columns
    assert "question_asker_intent_understanding" in sub_df.columns

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
