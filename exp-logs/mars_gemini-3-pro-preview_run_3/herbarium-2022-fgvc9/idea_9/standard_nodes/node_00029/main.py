import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader

# Import provided libraries
from library.utils import seed_everything, get_logger
from library.dataset import get_hierarchy_mappings, get_transforms, PlantDataset
from library.model import CascadingPlantModel
from library.engine import train_one_epoch, validate, generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    SEED = 42
    seed_everything(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("training.log")

    # Paths
    TRAIN_META_PATH = "./metadata/train.csv"
    VAL_META_PATH = "./metadata/val.csv"
    TEST_META_PATH = "./metadata/test.csv"
    HIERARCHY_JSON = "./input/train_metadata.json"
    CACHE_DIR = "./working/cache"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    INPUT_DIR = "./input"

    # Hyperparameters
    # Using 400k samples (approx 75% of data) to ensure high score while staying within time limits
    MAX_TRAIN_SAMPLES = 400000
    BATCH_SIZE_P1 = 128
    BATCH_SIZE_P2 = 64
    EPOCHS_P1 = 3
    EPOCHS_P2 = 2
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    logger.info(f"Device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    logger.info("Loading metadata...")
    df_train_full = pd.read_csv(TRAIN_META_PATH)
    df_val = pd.read_csv(VAL_META_PATH)
    df_test = pd.read_csv(TEST_META_PATH)

    # Hierarchy Mappings
    logger.info("Getting hierarchy mappings...")
    hierarchy_df, num_families, num_genera, num_species = get_hierarchy_mappings(
        HIERARCHY_JSON, load_cached_data=True, cache_dir=CACHE_DIR
    )
    logger.info(
        f"Hierarchy: {num_families} Families, {num_genera} Genera, {num_species} Species"
    )

    # Subsample training data
    if len(df_train_full) > MAX_TRAIN_SAMPLES:
        logger.info(f"Subsampling training data to {MAX_TRAIN_SAMPLES} samples...")
        df_train = df_train_full.sample(
            n=MAX_TRAIN_SAMPLES, random_state=SEED
        ).reset_index(drop=True)
    else:
        df_train = df_train_full

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    logger.info("Initializing CascadingPlantModel (EfficientNetV2-B0)...")
    model = CascadingPlantModel(
        num_species=num_species,
        num_genera=num_genera,
        num_families=num_families,
        backbone_name="tf_efficientnetv2_b0",
        pretrained=True,
        proj_dim=512,
    )
    model.to(DEVICE)

    # -------------------------------------------------------------------------
    # 4. Training Phase 1: 224x224
    # -------------------------------------------------------------------------
    logger.info("==== Phase 1: Training at 224x224 ====")

    train_transform_p1 = get_transforms("train", image_size=224)
    train_dataset_p1 = PlantDataset(
        df_train, INPUT_DIR, hierarchy_df, train_transform_p1
    )
    train_loader_p1 = DataLoader(
        train_dataset_p1,
        batch_size=BATCH_SIZE_P1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps_p1 = len(train_loader_p1) * EPOCHS_P1
    scheduler_p1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps_p1, pct_start=0.3
    )

    for epoch in range(1, EPOCHS_P1 + 1):
        logger.info(f"Phase 1 Epoch {epoch}/{EPOCHS_P1}")
        train_one_epoch(model, train_loader_p1, optimizer, DEVICE, epoch, scheduler_p1)

    # -------------------------------------------------------------------------
    # 5. Training Phase 2: 288x288
    # -------------------------------------------------------------------------
    logger.info("==== Phase 2: Training at 288x288 ====")

    train_transform_p2 = get_transforms("train", image_size=288)
    train_dataset_p2 = PlantDataset(
        df_train, INPUT_DIR, hierarchy_df, train_transform_p2
    )
    train_loader_p2 = DataLoader(
        train_dataset_p2,
        batch_size=BATCH_SIZE_P2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Re-init optimizer with lower LR for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=LR / 5, weight_decay=WEIGHT_DECAY)
    total_steps_p2 = len(train_loader_p2) * EPOCHS_P2
    scheduler_p2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR / 5, total_steps=total_steps_p2, pct_start=0.3
    )

    for epoch in range(1, EPOCHS_P2 + 1):
        logger.info(f"Phase 2 Epoch {epoch}/{EPOCHS_P2}")
        train_one_epoch(model, train_loader_p2, optimizer, DEVICE, epoch, scheduler_p2)

    # -------------------------------------------------------------------------
    # 6. Final Validation
    # -------------------------------------------------------------------------
    logger.info("==== Final Validation ====")
    val_transform_final = get_transforms("valid", image_size=288)
    val_dataset_final = PlantDataset(
        df_val, INPUT_DIR, hierarchy_df, val_transform_final
    )
    val_loader_final = DataLoader(
        val_dataset_final,
        batch_size=BATCH_SIZE_P2 * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    macro_f1, val_loss = validate(model, val_loader_final, DEVICE)
    print(f"Final Validation Metric: {macro_f1}")

    # -------------------------------------------------------------------------
    # 7. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("==== Failure Analysis ====")
    # Analyze a subset of validation data
    ANALYSIS_SIZE = 2000
    analysis_df = df_val.sample(
        n=min(len(df_val), ANALYSIS_SIZE), random_state=SEED
    ).reset_index(drop=True)
    analysis_dataset = PlantDataset(
        analysis_df, INPUT_DIR, hierarchy_df, val_transform_final
    )
    analysis_loader = DataLoader(
        analysis_dataset, batch_size=BATCH_SIZE_P2 * 2, shuffle=False, num_workers=4
    )

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in analysis_loader:
            images = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)

            logits, _, _ = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    errors = (all_preds != all_targets).astype(int)

    # Collect features
    widths = []
    heights = []
    file_sizes = []

    for idx, row in analysis_df.iterrows():
        path = os.path.join(INPUT_DIR, row["file_path"])
        try:
            file_sizes.append(os.path.getsize(path))
            img = cv2.imread(path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        except:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    # Compute correlations
    if len(errors) == len(file_sizes):
        corr_size = np.corrcoef(errors, file_sizes)[0, 1]
        corr_width = np.corrcoef(errors, widths)[0, 1]
        corr_height = np.corrcoef(errors, heights)[0, 1]

        print(f"Correlation (Error vs File Size): {corr_size}")
        print(f"Correlation (Error vs Width): {corr_width}")
        print(f"Correlation (Error vs Height): {corr_height}")

    # -------------------------------------------------------------------------
    # 8. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.5930838412243743
    if macro_f1 > THRESHOLD:
        logger.info("Validation metric passed threshold. Generating submission...")

        test_transform = get_transforms("test", image_size=288)
        test_dataset = PlantDataset(
            df_test,
            INPUT_DIR,
            hierarchy_df=None,
            transform=test_transform,
            is_test=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_P2 * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        generate_submission(model, test_loader, DEVICE, save_path=SUBMISSION_PATH)
    else:
        logger.info(
            f"Validation metric {macro_f1} did not beat threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
