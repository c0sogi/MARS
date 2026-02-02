import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

# Import pre-implemented components from the provided library
from library.utils import (
    set_seed,
    get_device,
    process_data,
    IcebergDataset,
    BHA_ResNet,
    train_one_fold,
    predict_test,
)


def run_pipeline(
    base_dir="./working/idea_31",
    submission_dir="./submission",
    epochs=75,
    patience=12,
    batch_size=32,
    lr=1e-3,
    n_folds=5,
):
    """
    Orchestrates the training of the BHA-ResNet using 5-Fold CV and generates the submission.
    """
    # 1. Environment Setup
    set_seed(42)
    device = get_device()
    print(f"Running pipeline on device: {device}")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Data Loading & Processing
    # process_data handles loading raw JSONs, processing images/angles, and caching.
    print("Preparing data...")
    X_train, y_train, angle_train, X_test, ids_test, angle_test = process_data(
        load_cached_data=True, base_dir=base_dir
    )

    # 3. Define Transforms
    # Apply random flips for training data augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    # 4. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    trained_models = []
    fold_best_losses = []

    print(f"Starting {n_folds}-Fold Cross-Validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n=== Fold {fold_idx} ===")

        # Split data for this fold
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        angle_tr, angle_val = angle_train[train_idx], angle_train[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, angle_tr, y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_val, angle_val, y_val, transform=None)

        # Create DataLoaders
        # Pin memory if using CUDA for faster transfer
        use_pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=use_pin_memory,
        )

        # Initialize Model
        model = BHA_ResNet().to(device)

        # Train Model
        best_state, best_loss = train_one_fold(
            fold_idx=fold_idx,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            patience=patience,
            lr=lr,
        )

        fold_best_losses.append(best_loss)

        # Load best weights and store model for ensemble inference
        model.load_state_dict(best_state)
        model.eval()
        trained_models.append(model)

    print(f"\nCV Complete. Losses per fold: {fold_best_losses}")
    print(f"Average CV Loss: {np.mean(fold_best_losses):.10f}")

    # 5. Inference on Test Set
    print("\nGenerating predictions for test set...")
    test_ds = IcebergDataset(X_test, angle_test, y=None, transform=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Get averaged predictions from the ensemble
    avg_preds = predict_test(trained_models, test_loader, device)

    # 6. Generate Submission File
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    sub_path = os.path.join(submission_dir, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


# Execute the pipeline
run_pipeline()
