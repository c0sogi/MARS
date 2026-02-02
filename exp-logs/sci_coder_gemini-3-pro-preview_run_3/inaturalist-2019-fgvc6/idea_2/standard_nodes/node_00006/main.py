import os
import sys
import torch
import numpy as np
import pandas as pd
from library import config
from library import utils
from library import dataset
from library import model
from library import engine


def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    utils.seed_everything()
    logger = utils.get_logger("runfile")
    device = utils.get_device()
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Load Metadata
    # -------------------------------------------------------------------------
    # Load metadata for train, validation, and test sets
    train_df, val_df, test_df = dataset.load_metadata()
    logger.info(
        f"Metadata loaded. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    logger.info("Initializing model...")
    net = model.create_model(num_classes=config.NUM_CLASSES, pretrained=True)
    net = net.to(device)

    # -------------------------------------------------------------------------
    # 4. Stage 1: Representation Learning (Low Res)
    # -------------------------------------------------------------------------
    # We reduce epochs slightly from config to ensure execution within time limits
    s1_config = config.STAGE_1_CONFIG.copy()
    s1_config["epochs"] = 3

    logger.info(f"Preparing data for {s1_config['stage_name']}...")
    train_loader_s1 = dataset.get_dataloader(
        train_df,
        image_size=s1_config["image_size"],
        batch_size=s1_config["batch_size"],
        is_training=True,
        sampling_strategy=s1_config["sampling_strategy"],
        rand_augment_config=s1_config,
    )
    val_loader_s1 = dataset.get_dataloader(
        val_df,
        image_size=s1_config["image_size"],
        batch_size=s1_config["batch_size"],
        is_training=False,
    )

    logger.info(f"Running {s1_config['stage_name']}...")
    net = engine.run_stage(net, train_loader_s1, val_loader_s1, s1_config)

    # -------------------------------------------------------------------------
    # 5. Stage 2: High-Resolution Adaptation
    # -------------------------------------------------------------------------
    s2_config = config.STAGE_2_CONFIG.copy()
    s2_config["epochs"] = 2

    logger.info(f"Preparing data for {s2_config['stage_name']}...")
    train_loader_s2 = dataset.get_dataloader(
        train_df,
        image_size=s2_config["image_size"],
        batch_size=s2_config["batch_size"],
        is_training=True,
        sampling_strategy=s2_config["sampling_strategy"],
        rand_augment_config=s2_config,
    )
    val_loader_s2 = dataset.get_dataloader(
        val_df,
        image_size=s2_config["image_size"],
        batch_size=s2_config["batch_size"],
        is_training=False,
    )

    logger.info(f"Running {s2_config['stage_name']}...")
    net = engine.run_stage(net, train_loader_s2, val_loader_s2, s2_config)

    # -------------------------------------------------------------------------
    # 6. Stage 3: Decoupled Classifier Alignment
    # -------------------------------------------------------------------------
    s3_config = config.STAGE_3_CONFIG.copy()
    s3_config["epochs"] = 2

    logger.info("Freezing backbone for Stage 3...")
    model.set_backbone_trainable(net, trainable=False)

    logger.info(f"Preparing data for {s3_config['stage_name']}...")
    train_loader_s3 = dataset.get_dataloader(
        train_df,
        image_size=s3_config["image_size"],
        batch_size=s3_config["batch_size"],
        is_training=True,
        sampling_strategy=s3_config["sampling_strategy"],
        rand_augment_config=s3_config,
    )
    # Reuse val_loader_s2 as image size is the same (384)

    logger.info(f"Running {s3_config['stage_name']}...")
    net = engine.run_stage(net, train_loader_s3, val_loader_s2, s3_config)

    # -------------------------------------------------------------------------
    # 7. Final Validation & Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Starting Final Validation and Failure Analysis...")
    net.eval()

    all_preds = []
    all_targets = []

    # Use val_loader_s2 (384px) for final evaluation
    with torch.no_grad():
        for images, targets, _ in val_loader_s2:
            images = images.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metric: Top-1 Error
    accuracy = (all_preds == all_targets).mean()
    top1_error = 1.0 - accuracy

    print(f"Final Validation Metric: {top1_error}")

    # Failure Analysis: Correlation between Error and Class Frequency
    # Calculate class counts from training data
    class_counts = train_df["category_id"].value_counts().to_dict()

    # Map counts to validation samples
    val_class_counts = [class_counts.get(t, 0) for t in all_targets]

    # Error vector (1 if error, 0 if correct)
    errors = (all_preds != all_targets).astype(int)

    # Compute Correlation
    if len(set(errors)) > 1:
        # Using numpy for correlation to avoid extra dependencies
        corr_matrix = np.corrcoef(errors, val_class_counts)
        corr = corr_matrix[0, 1]
        print(f"Correlation between Error and Class Frequency: {corr:.4f}")
    else:
        print("Cannot compute correlation: All predictions are correct or all wrong.")

    # -------------------------------------------------------------------------
    # 8. Submission
    # -------------------------------------------------------------------------
    if top1_error < 0.26:
        logger.info("Metric condition met. Generating submission...")

        test_loader = dataset.get_dataloader(
            test_df, image_size=384, batch_size=32, is_training=False
        )

        submission_rows = []

        with torch.no_grad():
            for images, _, image_ids in test_loader:
                images = images.to(device)

                # Test Time Augmentation (TTA): Original + Horizontal Flip
                # 1. Original
                outputs_orig = net(images).softmax(dim=1)

                # 2. Flipped
                images_flipped = torch.flip(images, dims=[3])
                outputs_flip = net(images_flipped).softmax(dim=1)

                # Average
                outputs_avg = (outputs_orig + outputs_flip) / 2.0

                # Get top 5 predictions
                _, topk_indices = torch.topk(outputs_avg, k=5, dim=1)

                topk_indices = topk_indices.cpu().numpy()
                image_ids = image_ids.numpy()

                for img_id, indices in zip(image_ids, topk_indices):
                    # Format: "id,predicted" where predicted is space separated list of category IDs
                    pred_str = " ".join(map(str, indices))
                    submission_rows.append({"id": img_id, "predicted": pred_str})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        logger.info(f"Metric {top1_error} >= 0.26. Skipping submission.")


if __name__ == "__main__":
    main()
