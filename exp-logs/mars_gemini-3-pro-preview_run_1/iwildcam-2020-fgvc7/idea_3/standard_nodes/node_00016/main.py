import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Import provided library modules
from library import config, utils, dataset, model, loss, engine, inference, bbox_handler


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    print(f"Running on device: {device}")

    # 2. Data Preparation
    print("Loading metadata...")
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Subsample training data for fast baseline execution (approx 25% of data)
    # This ensures the code completes well within the 2-hour limit
    TRAIN_SUBSET_SIZE = 30000
    if len(train_df) > TRAIN_SUBSET_SIZE:
        print(
            f"Subsampling training data to {TRAIN_SUBSET_SIZE} samples for fast baseline..."
        )
        train_df = train_df.sample(
            n=TRAIN_SUBSET_SIZE, random_state=config.SEED
        ).reset_index(drop=True)

    # Save temporary subsampled metadata
    temp_train_path = os.path.join(config.WORKING_DIR, "temp_train_subset.csv")
    train_df.to_csv(temp_train_path, index=False)

    # Initialize BBoxHandler (loads cached MegaDetector boxes)
    print("Initializing BBoxHandler...")
    bbox_h = bbox_handler.BBoxHandler(load_cached_data=True)

    # Create Datasets
    # Note: Validation uses the full set as required for the final metric
    train_dataset = dataset.WildCamDataset(
        metadata_path=temp_train_path, mode="train", bbox_handler=bbox_h
    )
    val_dataset = dataset.WildCamDataset(
        metadata_path=config.VAL_METADATA_PATH, mode="val", bbox_handler=bbox_h
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing model: {config.MODEL_NAME}")
    net = model.get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        device=device,
    )

    # Loss Function
    criterion = loss.FocalLoss(gamma=config.FOCAL_LOSS_GAMMA)

    # Optimizer
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    # Limit epochs for fast baseline
    BASELINE_EPOCHS = 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BASELINE_EPOCHS)

    # 4. Training
    print(f"Starting training for {BASELINE_EPOCHS} epochs...")
    engine.run_training(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=BASELINE_EPOCHS,
        device=device,
        save_path=config.BEST_MODEL_PATH,
    )

    # 5. Final Validation
    print("Loading best model for final evaluation...")
    if os.path.exists(config.BEST_MODEL_PATH):
        net.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found, using current weights.")

    net.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    print("Evaluating on full validation set...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Map batch back to image_ids for failure analysis
            # Since val_loader is not shuffled, we can index into val_df
            start_idx = i * config.BATCH_SIZE
            end_idx = start_idx + labels.size(0)
            batch_ids = val_df.iloc[start_idx:end_idx]["image_id"].values
            all_image_ids.extend(batch_ids)

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"Final Validation Metric: {final_acc}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    analysis_df = pd.DataFrame(
        {"image_id": all_image_ids, "label": all_labels, "pred": all_preds}
    )

    # Identify errors
    analysis_df["is_error"] = (analysis_df["label"] != analysis_df["pred"]).astype(int)

    # Merge with MegaDetector confidence from BBoxHandler cache
    # bbox_h.df has columns: ['image_id', 'x', 'y', 'w', 'h', 'conf']
    md_df = bbox_h.df[["image_id", "conf"]]
    analysis_df = analysis_df.merge(md_df, on="image_id", how="left")

    # Fill missing confidence with 0 (implies no detection found)
    analysis_df["conf"] = analysis_df["conf"].fillna(0.0)

    # Calculate correlation
    if len(analysis_df) > 0:
        corr = analysis_df["is_error"].corr(analysis_df["conf"])
        print(f"Correlation between Error and MegaDetector Confidence: {corr}")
    else:
        print("Analysis DataFrame is empty.")

    # 7. Submission
    # Threshold defined in task requirements
    SUBMISSION_THRESHOLD = 0.9670154903145775

    if final_acc > SUBMISSION_THRESHOLD:
        print(
            f"Validation accuracy ({final_acc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        inference.run_inference(
            checkpoint_path=config.BEST_MODEL_PATH,
            output_path=config.SUBMISSION_FILE_PATH,
            device=device,
        )
    else:
        print(
            f"Validation accuracy ({final_acc}) does not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
