import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from library.config import (
    DEVICE,
    MODEL_CONFIGS,
    SUBMISSION_PATH,
    CACHE_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    ETA_MIN,
    EARLY_STOPPING_PATIENCE,
)
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.architectures import WideSEResNet, DenseNetBC
from library.engine import Trainer, predict_with_tta, generate_submission


def main():
    # 1. Setup
    set_seed(42)

    # Extended epochs to allow convergence with Mixup and AdamW
    RUN_EPOCHS = 75

    print(f"Running Optimized Training with {RUN_EPOCHS} epochs...")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    loaders = get_dataloaders(load_cached_data=True)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    test_ids = loaders["test_ids"]

    # 3. Train Model 1: WideSEResNet
    print("\n=== Training WideSEResNet ===")
    cfg_resnet = MODEL_CONFIGS["wide_se_resnet"]
    model_resnet = WideSEResNet(
        depth=cfg_resnet["depth"],
        widen_factor=cfg_resnet["widen_factor"],
        drop_rate=cfg_resnet["drop_rate"],
        stem_type=cfg_resnet["stem_type"],
        se_reduction=cfg_resnet["se_reduction"],
    ).to(DEVICE)

    optimizer_resnet = optim.AdamW(
        model_resnet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler_resnet = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_resnet, T_max=RUN_EPOCHS, eta_min=ETA_MIN
    )
    criterion = nn.BCEWithLogitsLoss()

    resnet_save_path = os.path.join(CACHE_DIR, "best_resnet.pth")
    trainer_resnet = Trainer(
        model=model_resnet,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_resnet,
        scheduler=scheduler_resnet,
        device=DEVICE,
        patience=EARLY_STOPPING_PATIENCE,
        save_path=resnet_save_path,
    )
    trainer_resnet.fit(num_epochs=RUN_EPOCHS)

    # 4. Train Model 2: DenseNetBC
    print("\n=== Training DenseNetBC ===")
    cfg_dense = MODEL_CONFIGS["densenet_bc"]
    model_dense = DenseNetBC(
        growth_rate=cfg_dense["growth_rate"],
        block_config=cfg_dense["block_config"],
        compression=cfg_dense["compression"],
        num_init_features=cfg_dense["num_init_features"],
        stem_type=cfg_dense["stem_type"],
        drop_rate=cfg_dense["drop_rate"],
    ).to(DEVICE)

    optimizer_dense = optim.AdamW(
        model_dense.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler_dense = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_dense, T_max=RUN_EPOCHS, eta_min=ETA_MIN
    )

    densenet_save_path = os.path.join(CACHE_DIR, "best_densenet.pth")
    trainer_dense = Trainer(
        model=model_dense,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_dense,
        scheduler=scheduler_dense,
        device=DEVICE,
        patience=EARLY_STOPPING_PATIENCE,
        save_path=densenet_save_path,
    )
    trainer_dense.fit(num_epochs=RUN_EPOCHS)

    # 5. Ensemble Validation & Metrics
    print("\n=== Ensemble Validation ===")
    # Load best weights
    model_resnet.load_state_dict(torch.load(resnet_save_path))
    model_dense.load_state_dict(torch.load(densenet_save_path))

    # Get predictions on validation set using TTA for maximum accuracy
    print("Generating TTA predictions for validation set...")
    preds_resnet = predict_with_tta(model_resnet, val_loader, DEVICE)
    preds_dense = predict_with_tta(model_dense, val_loader, DEVICE)

    # Simple averaging ensemble
    ensemble_preds = (preds_resnet + preds_dense) / 2.0

    # Extract targets and images for evaluation and failure analysis
    val_targets = []
    val_images_list = []

    # We iterate again to get raw data matching the order of TTA predictions
    # Note: val_loader has shuffle=False, so order is preserved.
    for images, labels in val_loader:
        val_targets.append(labels.numpy())
        val_images_list.append(images.numpy())

    val_targets = np.concatenate(val_targets)
    val_images = np.concatenate(val_images_list)

    final_auc = roc_auc_score(val_targets, ensemble_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_targets - ensemble_preds)

    # Calculate meta-features on the images
    # Images are (B, 3, 32, 32). We average across spatial dims (2, 3).
    # Brightness: Mean of all channels
    brightness = val_images.mean(axis=(1, 2, 3))
    # Contrast: Std of all channels
    contrast = val_images.std(axis=(1, 2, 3))
    # Red Channel Mean: Mean of channel 0
    r_mean = val_images[:, 0, :, :].mean(axis=(1, 2))

    # Calculate correlations
    corr_brightness, _ = pearsonr(errors, brightness)
    corr_contrast, _ = pearsonr(errors, contrast)
    corr_r, _ = pearsonr(errors, r_mean)

    print(f"Error Correlation with Brightness: {corr_brightness}")
    print(f"Error Correlation with Contrast: {corr_contrast}")
    print(f"Error Correlation with Red Channel Mean: {corr_r}")

    # 7. Submission
    THRESHOLD = 0.9999694825433514

    if final_auc > THRESHOLD:
        print(f"\nMetric {final_auc} > {THRESHOLD}. Generating submission...")
        generate_submission(
            models=[model_resnet, model_dense],
            test_loader=test_loader,
            test_ids=test_ids,
            output_path=SUBMISSION_PATH,
        )
    else:
        print(f"\nMetric {final_auc} <= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
