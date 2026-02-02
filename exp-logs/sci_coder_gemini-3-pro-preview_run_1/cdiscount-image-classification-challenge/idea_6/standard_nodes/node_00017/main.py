import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import HierarchyManager
from library.dataset import BSONDataset
from library.model import MultiLevelResNet
from library.train import HierarchicalLoss, train_one_epoch, validate, set_seed


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for fast baseline execution within 3 minutes
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 4  # Reduce overhead

    # Limits for fast execution
    TRAIN_LIMIT = 5000
    VAL_LIMIT = 2000
    SUBMISSION_THRESHOLD = 0.6306776302037904

    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    hierarchy_manager = HierarchyManager(load_cached_data=True)

    train_dataset = BSONDataset(
        metadata_path=Config.TRAIN_METADATA,
        bson_path=Config.TRAIN_BSON,
        split="train",
        limit_size=TRAIN_LIMIT,
    )

    val_dataset = BSONDataset(
        metadata_path=Config.VAL_METADATA,
        bson_path=Config.TRAIN_BSON,
        split="val",
        limit_size=VAL_LIMIT,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = MultiLevelResNet()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Adjust scheduler for the limited dataset size
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.BASE_LR,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    criterion = HierarchicalLoss(
        hierarchy_manager=hierarchy_manager,
        device=device,
        label_smoothing=Config.LABEL_SMOOTHING,
    )

    scaler = GradScaler()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_val_acc = 0.0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)

    # Required Output
    print(f"Final Validation Metric: {best_val_acc}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    # We analyze correlation between error and number of images provided
    model.eval()
    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            preds = model(images, mask)
            logits_l3 = preds[0]
            _, predicted = torch.max(logits_l3, 1)

            # Calculate error (0 for correct, 1 for incorrect)
            is_incorrect = (predicted != targets).cpu().numpy().astype(int)

            # Calculate number of images (sum of mask)
            num_images = mask.sum(dim=1).cpu().numpy()

            for err, n_img in zip(is_incorrect, num_images):
                analysis_data.append({"error": err, "num_images": n_img})

    df_analysis = pd.DataFrame(analysis_data)
    if not df_analysis.empty and df_analysis["error"].std() > 0:
        correlation = df_analysis["error"].corr(df_analysis["num_images"])
        print(f"Correlation between Error and Num_Images: {correlation:.6f}")
    else:
        print("Correlation between Error and Num_Images: 0.000000")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    if best_val_acc > SUBMISSION_THRESHOLD:
        print("Validation metric threshold passed. Generating submission...")

        # Load Test Data
        test_dataset = BSONDataset(
            metadata_path=Config.TEST_METADATA, bson_path=Config.TEST_BSON, split="test"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load best model
        model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT))
        model.eval()

        all_preds = []
        all_ids = []

        # Pre-fetch category mapping array for fast lookup
        # mapping_df is sorted by internal class index
        category_lookup = hierarchy_manager.mapping_df["category_id"].values

        with torch.no_grad():
            for batch in test_loader:
                images = batch["images"].to(device)
                mask = batch["mask"].to(device)
                sample_ids = batch["sample_id"]

                preds = model(images, mask)
                logits_l3 = preds[0]
                _, predicted_indices = torch.max(logits_l3, 1)

                all_preds.append(predicted_indices.cpu().numpy())
                all_ids.append(sample_ids.numpy())

        # Concatenate and Map
        all_preds = np.concatenate(all_preds)
        all_ids = np.concatenate(all_ids)

        final_category_ids = category_lookup[all_preds]

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"_id": all_ids, "category_id": final_category_ids}
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
