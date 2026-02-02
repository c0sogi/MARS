import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library components
from library.config import (
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    CACHE_DIR,
    DEVICE,
    NUM_WORKERS,
    BATCH_SIZE,
    NUM_CLASSES,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
    SUBMISSION_FILE_PATH,
)
from library.feature_engine import extract_dataset_features
from library.dataset import CachedFeatureDataset, get_class_weights
from library.model import AttentionClassifier, get_category_mapping
from library.trainer import run_training
from library.utils import save_submission, calculate_accuracy


def main():
    # ==========================================
    # 1. SETUP & REPRODUCIBILITY
    # ==========================================
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ==========================================
    # 2. DATA SUBSAMPLING (FAST BASELINE)
    # ==========================================
    # We subsample the dataset to ensure the entire pipeline (extraction + training)
    # completes within the 2-hour limit.
    # 500k samples is large enough for a good baseline but small enough to process fast.
    TRAIN_SUBSET_SIZE = 500000
    VAL_SUBSET_SIZE = 50000

    print(
        f"Creating data subsets (Train: {TRAIN_SUBSET_SIZE}, Val: {VAL_SUBSET_SIZE})..."
    )

    train_full = pd.read_csv(TRAIN_META_PATH)
    val_full = pd.read_csv(VAL_META_PATH)

    # Random sampling
    train_subset = train_full.sample(
        n=min(TRAIN_SUBSET_SIZE, len(train_full)), random_state=SEED
    )
    val_subset = val_full.sample(
        n=min(VAL_SUBSET_SIZE, len(val_full)), random_state=SEED
    )

    # Save temporary metadata files
    mini_train_path = os.path.join(WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(WORKING_DIR, "mini_val.csv")

    train_subset.to_csv(mini_train_path, index=False)
    val_subset.to_csv(mini_val_path, index=False)

    # ==========================================
    # 3. FEATURE EXTRACTION
    # ==========================================
    # Extract features using frozen ResNet-50.
    # Batch size 256 is efficient for ResNet inference on A100.
    print("Extracting features for training subset...")
    train_feats, train_idx, train_labels, _ = extract_dataset_features(
        mini_train_path, TRAIN_BSON_PATH, CACHE_DIR, "mini_train", batch_size=256
    )

    print("Extracting features for validation subset...")
    val_feats, val_idx, val_labels, _ = extract_dataset_features(
        mini_val_path, TRAIN_BSON_PATH, CACHE_DIR, "mini_val", batch_size=256
    )

    # ==========================================
    # 4. DATASET & DATALOADER SETUP
    # ==========================================
    # Get category mapping (Category ID -> Index 0..N-1)
    cat2idx, idx2cat = get_category_mapping()

    def map_labels(labels_arr):
        # Map labels to indices, default to 0 if unknown (should not happen with correct metadata)
        return np.array([cat2idx.get(l, 0) for l in labels_arr])

    train_y = map_labels(train_labels)
    val_y = map_labels(val_labels)

    # Compute Class Weights to handle imbalance in the subset
    class_weights = get_class_weights(
        train_y, NUM_CLASSES, cache_path=None, load_cached_data=False
    )
    class_weights = class_weights.to(DEVICE)

    # Create Datasets
    train_ds = CachedFeatureDataset(train_feats, train_idx, train_y)
    val_ds = CachedFeatureDataset(val_feats, val_idx, val_y)

    # Create Loaders (Training on features allows large batch size)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # ==========================================
    # 5. MODEL TRAINING
    # ==========================================
    print("Initializing model...")
    model = AttentionClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5 Epochs is sufficient for convergence on pre-computed features
    EPOCHS = 5
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
    )

    model_save_path = os.path.join(WORKING_DIR, "baseline_model.pth")

    print("Starting training...")
    run_training(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=EPOCHS,
        patience=2,
        save_path=model_save_path,
        device=DEVICE,
    )

    # ==========================================
    # 6. VALIDATION & FAILURE ANALYSIS
    # ==========================================
    print("Performing final validation...")
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_preds = []
    all_targets = []
    all_num_imgs = []

    with torch.no_grad():
        for bags, masks, targets in val_loader:
            bags, masks, targets = bags.to(DEVICE), masks.to(DEVICE), targets.to(DEVICE)
            outputs = model(bags, masks)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Mask sum indicates the number of valid images in the bag
            all_num_imgs.extend(masks.sum(dim=1).cpu().numpy())

    final_acc = calculate_accuracy(all_preds, all_targets)
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    all_num_imgs_np = np.array(all_num_imgs)

    # Error indicator (1 if wrong, 0 if correct)
    errors = (all_preds_np != all_targets_np).astype(int)

    # Calculate correlation
    if np.std(errors) > 1e-9 and np.std(all_num_imgs_np) > 1e-9:
        corr = np.corrcoef(errors, all_num_imgs_np)[0, 1]
        print(f"Correlation between Error and Num_Images: {corr:.4f}")
    else:
        print("Correlation between Error and Num_Images: 0.0000")

    # ==========================================
    # 7. SUBMISSION
    # ==========================================
    THRESHOLD = 0.50636

    if final_acc > THRESHOLD:
        print(
            "Validation metric satisfactory. Generating submission for full test set..."
        )

        # Extract features for the FULL test set
        # This is the most time-consuming part but required for submission
        test_feats, test_idx, _, test_ids = extract_dataset_features(
            TEST_META_PATH, TEST_BSON_PATH, CACHE_DIR, "test_full", batch_size=256
        )

        test_ds = CachedFeatureDataset(test_feats, test_idx, labels=None)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        test_preds_idx = []
        with torch.no_grad():
            for bags, masks in test_loader:
                bags, masks = bags.to(DEVICE), masks.to(DEVICE)
                outputs = model(bags, masks)
                preds = torch.argmax(outputs, dim=1)
                test_preds_idx.extend(preds.cpu().numpy())

        # Map indices back to original Category IDs
        final_test_preds = [idx2cat[p] for p in test_preds_idx]

        save_submission(test_ids, final_test_preds, SUBMISSION_FILE_PATH)
    else:
        print(
            f"Validation metric {final_acc} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
