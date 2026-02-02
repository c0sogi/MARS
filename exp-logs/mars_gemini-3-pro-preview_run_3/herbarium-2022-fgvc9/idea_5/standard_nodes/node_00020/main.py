import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger, get_hierarchy_dicts
from library.dataset import (
    get_dataloaders,
    PlantDataset,
    get_transforms,
    get_test_dataloader,
)
from library.model import HierarchicalEfficientNet
from library.train import run_training_stage
from library.inference import predict_tta


def main():
    # 1. Setup and Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # Create output directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    logger = get_logger("RunFile", os.path.join(Config.OUTPUT_DIR, "run.log"))
    logger.info("Starting Fast Baseline Run...")

    # Override Config for speed (Fast Baseline requirements)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 30000  # Train on 30k samples
    Config.STAGE1_EPOCHS = 3  # Reduced epochs
    Config.STAGE2_EPOCHS = 2  # Reduced epochs

    device = torch.device(Config.DEVICE)

    # 2. Data Preparation
    # -------------------------------------------------------------------------
    logger.info("Generating/Loading Hierarchy Mappings...")
    # Ensure hierarchy mappings exist
    Config.get_hierarchy_mappings(load_cached_data=True)

    # 3. Model Initialization
    # -------------------------------------------------------------------------
    logger.info(f"Initializing Model: {Config.MODEL_NAME}")
    model = HierarchicalEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes_species=Config.NUM_CLASSES_SPECIES,
        num_classes_genus=Config.NUM_CLASSES_GENUS,
        num_classes_family=Config.NUM_CLASSES_FAMILY,
    )
    model = model.to(device)

    # 4. Training Stages
    # -------------------------------------------------------------------------

    # --- Stage 1: Low Res (224x224) ---
    logger.info("Retrieving Stage 1 DataLoaders...")
    train_loader_s1, val_loader_s1 = get_dataloaders(stage=1, debug=Config.DEBUG)

    logger.info("Running Stage 1 Training...")
    run_training_stage(
        stage_num=1,
        model=model,
        train_loader=train_loader_s1,
        val_loader=val_loader_s1,
        epochs=Config.STAGE1_EPOCHS,
        lr=Config.STAGE1_LR,
        device=device,
        logger=logger,
    )

    # --- Stage 2: High Res (320x320) ---
    logger.info("Retrieving Stage 2 DataLoaders...")
    # Note: run_training_stage reloads the best weights from the previous stage if saved,
    # but here we pass the model instance which keeps weights in memory.
    # Ideally we ensure we are continuing from the best state.
    # The provided run_training_stage loads the best model from disk at the end of the stage.

    train_loader_s2, val_loader_s2 = get_dataloaders(stage=2, debug=Config.DEBUG)

    logger.info("Running Stage 2 Training...")
    run_training_stage(
        stage_num=2,
        model=model,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        epochs=Config.STAGE2_EPOCHS,
        lr=Config.STAGE2_LR,
        device=device,
        logger=logger,
    )

    # 5. Full Validation & Metrics
    # -------------------------------------------------------------------------
    logger.info("Performing Full Validation on Hold-out Set...")

    # Construct Full Validation Loader (ignoring DEBUG flag to meet requirement)
    df_val = pd.read_csv(Config.VAL_CSV)
    hierarchy_dicts = get_hierarchy_dicts(load_cached_data=True)

    val_dataset_full = PlantDataset(
        df=df_val,
        transforms=get_transforms("val", Config.STAGE2_IMAGE_SIZE),
        hierarchy_dicts=hierarchy_dicts,
        is_test=False,
    )

    val_loader_full = DataLoader(
        val_dataset_full,
        batch_size=Config.STAGE2_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader_full:
            images = images.to(device)
            # We only care about species for the primary metric
            true_labels = targets["species"].numpy()

            outputs = model(images)
            species_logits = outputs["species"]
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(true_labels)

    # Calculate Metric
    final_f1 = f1_score(all_targets, all_preds, average="macro")

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_f1}")

    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Calculate per-class F1
    class_f1_scores = f1_score(all_targets, all_preds, average=None)
    unique_classes = np.unique(all_targets)

    # Load training data to get class counts (Input Feature: Class Frequency)
    df_train = pd.read_csv(Config.TRAIN_CSV)
    class_counts = df_train["category_id"].value_counts().to_dict()

    # Align counts with F1 scores
    counts_aligned = []
    f1_aligned = []

    # Get mapping to convert label back to category_id
    label_to_species = hierarchy_dicts[3]

    for cls_label, score in zip(unique_classes, class_f1_scores):
        # Map label to category_id
        cls_id = label_to_species[cls_label]

        # Some classes might be in val but not train (unlikely due to stratification)
        # or vice versa.
        cnt = class_counts.get(cls_id, 0)
        counts_aligned.append(cnt)
        f1_aligned.append(score)

    if len(counts_aligned) > 1:
        correlation = np.corrcoef(counts_aligned, f1_aligned)[0, 1]
        print(
            f"Correlation between Class Training Frequency and Validation F1 Score: {correlation}"
        )
    else:
        print("Insufficient data for correlation analysis.")

    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.5930838412243743

    if final_f1 > THRESHOLD:
        logger.info(f"Metric {final_f1} > {THRESHOLD}. Generating Submission...")

        # Use Stage 2 image size for inference
        test_loader = get_test_dataloader(
            Config.STAGE2_IMAGE_SIZE, Config.STAGE2_BATCH_SIZE
        )

        # Run TTA Inference
        df_submission = predict_tta(model, test_loader, device)

        # Save to required path
        submission_path = "./submission/submission.csv"
        df_submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    else:
        logger.info(f"Metric {final_f1} <= {THRESHOLD}. Skipping Submission.")


if __name__ == "__main__":
    main()
