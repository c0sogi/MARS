import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import library modules
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    SUBMISSION_PATH,
    CHECKPOINT_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    LR,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    ENSEMBLE_CONFIG,
    SWA_LR,
    PATIENCE,
    CALIBRATION_EPOCHS,
    SWA_EPOCHS,
)
from library.utils import set_seed, setup_logger, calculate_log_loss
from library.data import (
    load_and_process_data,
    IcebergDataset,
    get_transforms,
    compute_normalization_stats,
)
from library.model import IcebergResNet
from library.engine import (
    train_one_epoch,
    validate_tta,
    run_swa_phase,
    predict_ensemble,
)


def main():
    # 1. Setup
    set_seed(SEED)
    logger = setup_logger("baseline", os.path.join(CHECKPOINT_DIR, "train.log"))
    logger.info("Starting Hybrid-Input SWA-ResNet Pipeline")

    # 2. Data Loading
    # Load raw data (cached)
    train_data_full, test_data_full = load_and_process_data(load_cached_data=True)

    # Load metadata
    df_train_meta = pd.read_csv(TRAIN_META_PATH)
    df_val_meta = pd.read_csv(VAL_META_PATH)

    # Helper to subset data based on IDs in metadata
    def get_subset(full_data, ids_list):
        id_map = {id_: i for i, id_ in enumerate(full_data["ids"])}
        indices = [id_map[id_] for id_ in ids_list if id_ in id_map]
        return {k: v[indices] for k, v in full_data.items()}

    train_subset = get_subset(train_data_full, df_train_meta["id"].values)
    val_subset = get_subset(train_data_full, df_val_meta["id"].values)

    # Compute stats on training subset only to avoid leakage
    stats = compute_normalization_stats(train_subset["band_1"], train_subset["band_2"])
    angle_stats = {
        "mean": np.nanmean(train_subset["inc_angle"]),
        "std": np.nanstd(train_subset["inc_angle"]) + 1e-6,
    }

    logger.info(f"Train subset size: {len(train_subset['ids'])}")
    logger.info(f"Val subset size: {len(val_subset['ids'])}")

    # 3. Phase 1: Calibration (Finding convergence epoch)
    logger.info("Phase 1: Calibration (Finding convergence epoch)")

    n_folds = 5
    epochs_p1 = CALIBRATION_EPOCHS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_losses = np.zeros((n_folds, epochs_p1))
    final_lrs = []

    X_indices = np.arange(len(train_subset["ids"]))
    y_train = train_subset["labels"]

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_indices, y_train)):
        logger.info(f"Calibrating Fold {fold+1}/{n_folds}")

        # Prepare Fold Data
        fold_train = {k: v[t_idx] for k, v in train_subset.items()}
        fold_val = {k: v[v_idx] for k, v in train_subset.items()}

        # Datasets (Use Variant A for calibration)
        ds_train = IcebergDataset(
            fold_train["band_1"],
            fold_train["band_2"],
            fold_train["inc_angle"],
            fold_train["labels"],
            fold_train["ids"],
            variant="A",
            transform=get_transforms("train"),
            stats=stats,
            angle_stats=angle_stats,
        )
        ds_val = IcebergDataset(
            fold_val["band_1"],
            fold_val["band_2"],
            fold_val["inc_angle"],
            fold_val["labels"],
            fold_val["ids"],
            variant="A",
            transform=get_transforms("valid"),
            stats=stats,
            angle_stats=angle_stats,
        )

        dl_train = DataLoader(
            ds_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Model & Opt
        model = IcebergResNet().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=PATIENCE
        )

        for epoch in range(epochs_p1):
            _ = train_one_epoch(
                model,
                dl_train,
                optimizer,
                DEVICE,
                epoch,
                label_smoothing=LABEL_SMOOTHING,
            )
            val_loss = validate_tta(model, dl_val, DEVICE)
            scheduler.step(val_loss)
            fold_losses[fold, epoch] = val_loss

        final_lrs.append(optimizer.param_groups[0]["lr"])

    # Analyze Phase 1 Results
    avg_losses = fold_losses.mean(axis=0)
    best_epoch = int(np.argmin(avg_losses)) + 1  # 1-based index
    avg_final_lr = float(np.mean(final_lrs))

    logger.info(
        f"Calibration Complete. Best Epoch: {best_epoch}, Final LR: {avg_final_lr:.2e}"
    )

    # 4. Phase 2: Production (Ensemble Training)
    logger.info("Phase 2: Production (Training Ensemble on full train subset)")

    trained_models = []

    # Helper to create full train loader for a specific variant
    def get_full_loader(variant):
        ds = IcebergDataset(
            train_subset["band_1"],
            train_subset["band_2"],
            train_subset["inc_angle"],
            train_subset["labels"],
            train_subset["ids"],
            variant=variant,
            transform=get_transforms("train"),
            stats=stats,
            angle_stats=angle_stats,
        )
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    for i, config in enumerate(ENSEMBLE_CONFIG):
        variant = config["variant"]
        seed_val = config["seed"]
        set_seed(seed_val)

        logger.info(f"Training Model {i+1}/{len(ENSEMBLE_CONFIG)} (Variant {variant})")

        loader = get_full_loader(variant)
        model = IcebergResNet().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # Schedule Mapping: Cosine Annealing to discovered convergence point
        scheduler = CosineAnnealingLR(optimizer, T_max=best_epoch, eta_min=avg_final_lr)

        # Main Training
        for epoch in range(best_epoch):
            train_one_epoch(
                model, loader, optimizer, DEVICE, epoch, label_smoothing=LABEL_SMOOTHING
            )
            scheduler.step()

        # SWA Phase
        swa_epochs = SWA_EPOCHS
        swa_model = run_swa_phase(
            model,
            loader,
            optimizer,
            DEVICE,
            swa_epochs,
            SWA_LR,
            label_smoothing=LABEL_SMOOTHING,
        )

        trained_models.append(swa_model)

    # 5. Validation on Hold-out Set
    logger.info("Evaluating on Hold-out Validation Set...")

    # Create Val Datasets for A and B
    ds_val_A = IcebergDataset(
        val_subset["band_1"],
        val_subset["band_2"],
        val_subset["inc_angle"],
        val_subset["labels"],
        val_subset["ids"],
        variant="A",
        transform=get_transforms("valid"),
        stats=stats,
        angle_stats=angle_stats,
    )
    ds_val_B = IcebergDataset(
        val_subset["band_1"],
        val_subset["band_2"],
        val_subset["inc_angle"],
        val_subset["labels"],
        val_subset["ids"],
        variant="B",
        transform=get_transforms("valid"),
        stats=stats,
        angle_stats=angle_stats,
    )

    dl_val_A = DataLoader(
        ds_val_A, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    dl_val_B = DataLoader(
        ds_val_B, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    val_ids = val_subset["ids"]
    val_preds_matrix = np.zeros((len(val_ids), len(trained_models)))
    val_id_map = {id_: i for i, id_ in enumerate(val_ids)}

    # Manual Prediction Loop for Validation (to handle labels in dataset)
    for i, (model, config) in enumerate(zip(trained_models, ENSEMBLE_CONFIG)):
        variant = config["variant"]
        loader = dl_val_A if variant == "A" else dl_val_B

        model.eval()
        preds = []
        ids_list = []
        with torch.no_grad():
            for images, angles, targets, ids in loader:
                images = images.to(DEVICE)
                angles = angles.to(DEVICE)

                # TTA Logic
                images_h = torch.flip(images, [3])
                images_v = torch.flip(images, [2])
                images_hv = torch.flip(images, [2, 3])

                p1 = torch.sigmoid(model(images, angles))
                p2 = torch.sigmoid(model(images_h, angles))
                p3 = torch.sigmoid(model(images_v, angles))
                p4 = torch.sigmoid(model(images_hv, angles))

                avg_p = (p1 + p2 + p3 + p4) / 4.0
                preds.extend(avg_p.cpu().numpy().flatten())
                ids_list.extend(ids)

        for pid, pprob in zip(ids_list, preds):
            if pid in val_id_map:
                val_preds_matrix[val_id_map[pid], i] = pprob

    final_val_preds = val_preds_matrix.mean(axis=1)
    final_val_loss = calculate_log_loss(val_subset["labels"], final_val_preds)

    print(f"Final Validation Metric: {final_val_loss}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    errors = np.abs(val_subset["labels"] - final_val_preds)

    angles = val_subset["inc_angle"]
    valid_mask = ~np.isnan(angles)
    if np.sum(valid_mask) > 0:
        corr = np.corrcoef(errors[valid_mask], angles[valid_mask])[0, 1]
        print(f"Correlation between Error and Incidence Angle: {corr:.4f}")
    else:
        print("Correlation between Error and Incidence Angle: NaN")

    # 7. Submission
    threshold = 0.16918645240183008
    if final_val_loss < threshold:
        logger.info(f"Metric {final_val_loss} < {threshold}. Generating Submission...")

        ds_test_A = IcebergDataset(
            test_data_full["band_1"],
            test_data_full["band_2"],
            test_data_full["inc_angle"],
            None,
            test_data_full["ids"],
            variant="A",
            transform=get_transforms("test"),
            stats=stats,
            angle_stats=angle_stats,
        )
        ds_test_B = IcebergDataset(
            test_data_full["band_1"],
            test_data_full["band_2"],
            test_data_full["inc_angle"],
            None,
            test_data_full["ids"],
            variant="B",
            transform=get_transforms("test"),
            stats=stats,
            angle_stats=angle_stats,
        )

        dl_test_A = DataLoader(
            ds_test_A, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        dl_test_B = DataLoader(
            ds_test_B, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        test_ids = test_data_full["ids"]
        test_preds_matrix = np.zeros((len(test_ids), len(trained_models)))
        test_id_map = {id_: i for i, id_ in enumerate(test_ids)}

        for i, (model, config) in enumerate(zip(trained_models, ENSEMBLE_CONFIG)):
            variant = config["variant"]
            loader = dl_test_A if variant == "A" else dl_test_B

            # predict_ensemble works for datasets without labels (returns 3 items)
            ids, probs = predict_ensemble([model], loader, DEVICE)

            for pid, pprob in zip(ids, probs):
                if pid in test_id_map:
                    test_preds_matrix[test_id_map[pid], i] = pprob

        final_test_preds = test_preds_matrix.mean(axis=1)

        submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": final_test_preds})
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {SUBMISSION_PATH}")
    else:
        logger.info(f"Metric {final_val_loss} >= {threshold}. Skipping Submission.")


if __name__ == "__main__":
    main()
