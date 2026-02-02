import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided libraries
from library import config
from library import utils
from library import dataset
from library import model
from library import train


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Preparation
    # We use the full dataset (debug=False) to ensuring valid class coverage,
    # but we will limit the training epochs to 1 to meet the "fast baseline" constraint.
    debug_mode = False

    # Get mappings
    print("Loading label mappings...")
    label2id, id2label = dataset.get_label_mapping(load_cached_data=True)

    # Initialize Datasets
    print("Initializing datasets...")
    train_ds = dataset.PlantDataset(
        split="train", transform=dataset.get_transforms("train"), debug=debug_mode
    )
    val_ds = dataset.PlantDataset(
        split="val", transform=dataset.get_transforms("val"), debug=debug_mode
    )
    test_ds = dataset.PlantDataset(
        split="test", transform=dataset.get_transforms("test"), debug=debug_mode
    )

    # Initialize DataLoaders
    print("Initializing dataloaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    # 3. Model Initialization
    print("Initializing model...")
    net = model.PlantClassifier(pretrained=True)
    net.to(device)

    # Calculate class weights to address prior bias (Cite {solution_lesson_node_00001})
    print("Calculating class weights...")
    train_df = pd.read_csv(config.TRAIN_META_PATH)
    # Map category_id to count
    cat_counts = train_df["category_id"].value_counts().to_dict()

    # Create weights array aligned with label2id (0..NUM_CLASSES-1)
    weights = np.zeros(len(label2id), dtype=np.float32)
    for cat_id, label_idx in label2id.items():
        count = cat_counts.get(cat_id, 0)
        # Inverse square root weighting
        weights[label_idx] = 1.0 / np.sqrt(max(count, 1))

    # Normalize weights
    weights = weights / weights.mean()
    class_weights = torch.from_numpy(weights).float().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # 4. Training
    num_epochs = config.NUM_EPOCHS
    print(f"Starting training for {num_epochs} epoch(s)...")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
    )

    for epoch in range(num_epochs):
        train_loss = train.train_one_epoch(
            net, train_loader, criterion, optimizer, device, scheduler
        )
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        # Intermediate validation
        val_loss, val_f1 = train.validate(net, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

    # 5. Final Validation & Failure Analysis
    print("Performing final validation and failure analysis...")
    net.eval()

    all_preds = []
    all_labels = []

    # We manually iterate to get predictions for analysis
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Metric
    final_f1 = utils.calculate_metrics(all_labels, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    # Error magnitude: 0 for correct, 1 for incorrect
    errors = (all_preds != all_labels).astype(int)

    # Correlation with Region ID
    # val_ds.df contains the metadata and is aligned because shuffle=False
    if "region_id" in val_ds.df.columns:
        regions = val_ds.df["region_id"].values

        # Check lengths to ensure alignment
        if len(regions) == len(errors):
            # Calculate correlation
            corr = np.corrcoef(errors, regions)[0, 1]
            print(f"Correlation between Error and Region ID: {corr:.6f}")

            # Additional analysis: Error rate per region
            df_analysis = pd.DataFrame({"region": regions, "error": errors})
            print("Error rate by region:")
            print(df_analysis.groupby("region")["error"].mean())
        else:
            print(
                f"Warning: Metadata length ({len(regions)}) != Predictions length ({len(errors)}). Skipping correlation."
            )
    else:
        print("Region ID not found in metadata. Skipping correlation analysis.")

    # 6. Submission
    if final_f1 > 0.5413:
        print("Generating predictions for test set...")
        submission_df = train.inference(net, test_loader, device, id2label)

        # Save
        config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        save_path = config.SUBMISSION_DIR / "submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(f"Final F1 ({final_f1}) did not meet threshold. Skipping submission.")


if __name__ == "__main__":
    main()
