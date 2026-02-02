import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import (
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    DEVICE,
    NUM_WORKERS,
    BATCH_SIZE,
    SEED,
    WORKING_DIR,
    SUBMISSION_DIR,
)
from library.extract_features import (
    DualBackbone,
    ProductDataset,
    collate_fn,
    extract_features_from_loader,
    EXTRACTION_BATCH_SIZE,
)
from library.data_utils import HierarchyManager
from library.feature_dataset import get_dataloaders
from library.trainer import Trainer


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    print("Starting runfile.py execution...")

    # ==========================================
    # 1. PREPARE DATA METADATA
    # ==========================================
    print("Preparing data configuration...")

    # Define subset size for training (Fast Baseline)
    TRAIN_SUBSET_SIZE = 200000

    # Load full training metadata to sample from
    train_meta_full = pd.read_csv(TRAIN_META_PATH)

    # Create Train Subset
    train_subset = train_meta_full.sample(
        n=min(TRAIN_SUBSET_SIZE, len(train_meta_full)), random_state=SEED
    )
    train_subset_path = os.path.join(WORKING_DIR, "train_subset.csv")
    train_subset.to_csv(train_subset_path, index=False)
    print(f"Train subset prepared: {len(train_subset)} samples.")

    # We use full VAL and TEST metadata as required
    print(f"Using Full Validation Set: {VAL_META_PATH}")
    print(f"Using Full Test Set: {TEST_META_PATH}")

    # ==========================================
    # 2. FEATURE EXTRACTION
    # ==========================================
    print("Initializing Feature Extraction Model...")
    model = DualBackbone().to(DEVICE)
    model.eval()

    # Define paths for extracted features
    train_feats_path = os.path.join(WORKING_DIR, "train_feats_subset.npy")
    train_labels_path = os.path.join(WORKING_DIR, "train_labels_subset.npy")
    val_feats_path = os.path.join(WORKING_DIR, "val_feats_full.npy")
    val_labels_path = os.path.join(WORKING_DIR, "val_labels_full.npy")
    test_feats_path = os.path.join(WORKING_DIR, "test_feats_full.npy")
    test_ids_path = os.path.join(WORKING_DIR, "test_ids_full.npy")

    # --- Extract Train Subset ---
    if not os.path.exists(train_feats_path):
        print("Extracting features for Train Subset...")
        ds = ProductDataset(train_subset_path, TRAIN_BSON_PATH)
        loader = DataLoader(
            ds,
            batch_size=EXTRACTION_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        feats, lbls, _ = extract_features_from_loader(model, loader, DEVICE)
        np.save(train_feats_path, feats)
        np.save(train_labels_path, lbls)
    else:
        print("Train features already exist, skipping extraction.")

    # --- Extract Full Validation Set ---
    if not os.path.exists(val_feats_path):
        print("Extracting features for Full Validation Set...")
        ds = ProductDataset(VAL_META_PATH, TRAIN_BSON_PATH)
        loader = DataLoader(
            ds,
            batch_size=EXTRACTION_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        feats, lbls, _ = extract_features_from_loader(model, loader, DEVICE)
        np.save(val_feats_path, feats)
        np.save(val_labels_path, lbls)
    else:
        print("Validation features already exist, skipping extraction.")

    # --- Extract Full Test Set ---
    if not os.path.exists(test_feats_path):
        print("Extracting features for Full Test Set...")
        ds = ProductDataset(TEST_META_PATH, TEST_BSON_PATH)
        loader = DataLoader(
            ds,
            batch_size=EXTRACTION_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        feats, _, ids = extract_features_from_loader(model, loader, DEVICE)
        np.save(test_feats_path, feats)
        np.save(test_ids_path, ids)
    else:
        print("Test features already exist, skipping extraction.")

    # Clean up model to free GPU memory
    del model
    torch.cuda.empty_cache()

    # ==========================================
    # 3. TRAINING
    # ==========================================
    print("Initializing Training...")

    # Init Hierarchy Manager
    hm = HierarchyManager(load_cached_data=True)

    # Get DataLoaders
    # Note: We pass the subset paths for train, but full paths for val/test
    train_loader, val_loader, test_loader = get_dataloaders(
        train_features_path=train_feats_path,
        train_labels_path=train_labels_path,
        val_features_path=val_feats_path,
        val_labels_path=val_labels_path,
        test_features_path=test_feats_path,
        test_ids_path=test_ids_path,
        hierarchy_manager=hm,
        batch_size=BATCH_SIZE,
        mixup_alpha=0.2,
        num_workers=NUM_WORKERS,
    )

    # Init Trainer
    trainer = Trainer(hm)

    # Run Training
    # 10 epochs is sufficient for the MLP heads to converge on the subset
    print("Starting Training Loop...")
    trainer.fit(train_loader, val_loader, epochs=10, patience=3)

    # ==========================================
    # 4. VALIDATION & FAILURE ANALYSIS
    # ==========================================
    print("Performing Final Validation and Failure Analysis...")

    # Load best model
    trainer.model.load_state_dict(
        torch.load(trainer.model_save_path, map_location=DEVICE)
    )
    trainer.model.eval()

    all_errors = []
    all_norms = []

    correct_l3 = 0
    total = 0

    # Manual validation loop to collect metrics and analysis data on the FULL validation set
    with torch.no_grad():
        for features, targets_a, _, _ in val_loader:
            features = features.to(DEVICE)
            l3_target = targets_a[2].to(DEVICE)

            _, _, logits_l3 = trainer.model(features)
            preds_l3 = torch.argmax(logits_l3, dim=1)

            # Metric Calculation
            correct_l3 += (preds_l3 == l3_target).sum().item()
            total += features.size(0)

            # Failure Analysis Data
            # Error = 1 if wrong, 0 if correct
            errors = (preds_l3 != l3_target).float().cpu().numpy()
            # Feature Norm (L2)
            norms = torch.norm(features, p=2, dim=1).cpu().numpy()

            all_errors.extend(errors)
            all_norms.extend(norms)

    final_acc = correct_l3 / total
    print(f"Final Validation Metric: {final_acc}")

    # Correlation Analysis
    if len(all_errors) > 0:
        correlation = np.corrcoef(all_errors, all_norms)[0, 1]
        print(f"Correlation between Error and Feature Norm: {correlation}")
    else:
        print("Correlation between Error and Feature Norm: NaN")

    # ==========================================
    # 5. SUBMISSION
    # ==========================================
    THRESHOLD = 0.6239621493939094

    if final_acc > THRESHOLD:
        print(
            f"Validation accuracy {final_acc} > {THRESHOLD}. Generating submission..."
        )
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        trainer.predict(test_loader, output_csv_path=submission_path)
    else:
        print(f"Validation accuracy {final_acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
