import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import (
    SEED,
    DEVICE,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    N_FOLDS,
    INPUT_DIR,
    SUBMISSION_PATH,
)
from library.utils import seed_everything, get_device
from library.data_loader import generate_dataset_arrays, AGIVDataset, get_transforms
from library.network import build_model
from library.engine import train_model, predict


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading & Preparation
    print("Loading metadata and datasets...")

    # Load Metadata
    df_train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(TEST_METADATA_PATH)

    # Load/Generate Data Arrays (using caching)
    # We load both train and val splits as defined in metadata, then merge them for CV
    ids_train, imgs_train, tgts_train = generate_dataset_arrays(
        df_train_meta, "train", load_cached_data=True
    )
    ids_val, imgs_val, tgts_val = generate_dataset_arrays(
        df_val_meta, "val", load_cached_data=True
    )
    ids_test, imgs_test, _ = generate_dataset_arrays(
        df_test_meta, "test", load_cached_data=True
    )

    # Concatenate Train and Val for 5-Fold CV
    all_ids = np.concatenate([ids_train, ids_val])
    all_imgs = np.concatenate([imgs_train, imgs_val])
    all_tgts = np.concatenate([tgts_train, tgts_val])

    print(f"Total training samples (Train+Val): {len(all_ids)}")
    print(f"Total test samples: {len(ids_test)}")

    # Prepare Test Loader (Static across folds)
    test_dataset = AGIVDataset(imgs_test, None, transform=get_transforms("test"))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(all_tgts))
    test_preds_accumulator = np.zeros(len(ids_test))

    print(f"Starting {N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_imgs, all_tgts)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Split Data
        X_train, X_val = all_imgs[train_idx], all_imgs[val_idx]
        y_train, y_val = all_tgts[train_idx], all_tgts[val_idx]

        # Create Datasets
        train_ds = AGIVDataset(X_train, y_train, transform=get_transforms("train"))
        val_ds = AGIVDataset(X_val, y_val, transform=get_transforms("val"))

        # Create Loaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = build_model(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Train
        model = train_model(
            model, train_loader, val_loader, optimizer, device, NUM_EPOCHS, PATIENCE
        )

        # Inference on Validation (OOF)
        val_preds = predict(model, val_loader, device)
        oof_preds[val_idx] = val_preds

        # Inference on Test
        fold_test_preds = predict(model, test_loader, device)
        test_preds_accumulator += fold_test_preds

        # Cleanup to save memory
        del model, optimizer, train_loader, val_loader, X_train, X_val
        torch.cuda.empty_cache()

    # 4. Validation & Failure Analysis
    final_auc = roc_auc_score(all_tgts, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(all_tgts - oof_preds)

    # Generate metadata feature for correlation: T1wCE slice count
    # We need to reconstruct the dataframe order to match 'all_ids'
    # Since we concatenated train then val, we do the same for dataframes
    df_combined = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Verify alignment (sanity check)
    if not np.array_equal(df_combined["BraTS21ID"].values, all_ids):
        print("Warning: ID mismatch in failure analysis. Skipping correlation.")
    else:
        print("Computing T1wCE slice counts for failure analysis...")
        slice_counts = []
        for _, row in df_combined.iterrows():
            # Construct path: input/train/00xxx/T1wCE
            # The metadata path is relative to input/
            folder_path = os.path.join(INPUT_DIR, row["t1wce_path"])
            try:
                # Fast count of files
                count = len(
                    [name for name in os.listdir(folder_path) if name.endswith(".dcm")]
                )
            except Exception:
                count = 0
            slice_counts.append(count)

        df_combined["t1wce_count"] = slice_counts
        df_combined["error"] = errors

        # Calculate correlation
        corr = df_combined["error"].corr(df_combined["t1wce_count"])
        print(f"Correlation between Error and T1wCE Slice Count: {corr}")

    # 5. Submission
    avg_test_preds = test_preds_accumulator / N_FOLDS

    threshold = 0.6705454545454544
    if final_auc > threshold:
        print(
            f"Validation metric ({final_auc}) > threshold ({threshold}). Generating submission..."
        )

        df_sub = pd.DataFrame({"BraTS21ID": ids_test, "MGMT_value": avg_test_preds})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric ({final_auc}) failed to meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
