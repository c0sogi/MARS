import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.dataset import CervicalSpineDataset, get_transforms
from library.model import CervicalFractureNet
from library.loss import HybridLoss
from library.engine import train_one_epoch, validate
from library.utils import get_all_study_paths


def main():
    # 1. Setup
    Config.setup_reproducibility(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Using the full provided training set (161 samples) as it is small enough
    # to run quickly even without subsampling.

    # Datasets
    train_dataset = CervicalSpineDataset(
        train_df, mode="train", transform=get_transforms("train")
    )
    val_dataset = CervicalSpineDataset(
        val_df, mode="val", transform=get_transforms("val")
    )

    # DataLoaders
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
        drop_last=False,
    )

    # 3. Model & Optimization
    model = CervicalFractureNet()
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    loss_fn = HybridLoss()

    # 4. Training Loop
    best_metric = float("inf")
    best_model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, loss_fn, epoch
        )

        # Validate
        val_loss, metric, _ = validate(model, val_loader, device, loss_fn)

        # Checkpoint
        if metric < best_metric:
            best_metric = metric
            torch.save(model.state_dict(), best_model_path)

    print(f"Final Validation Metric: {best_metric}")

    # 5. Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get predictions on validation set
    _, _, val_preds_df = validate(model, val_loader, device, loss_fn)

    # Prepare Ground Truth for Analysis
    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    val_long = val_df.melt(
        id_vars=["StudyInstanceUID"],
        value_vars=target_cols,
        var_name="subtype",
        value_name="target",
    )
    val_long["row_id"] = val_long["StudyInstanceUID"] + "_" + val_long["subtype"]

    # Merge predictions with targets
    analysis_df = val_long.merge(val_preds_df, on="row_id", how="inner")

    # Calculate Absolute Error
    analysis_df["error"] = (analysis_df["target"] - analysis_df["fractured"]).abs()

    # Aggregate error per study
    study_error = analysis_df.groupby("StudyInstanceUID")["error"].mean().reset_index()

    # Get Metadata Features (Slice Count)
    # We use the cached path utility on the training directory (where val images also reside)
    paths_dict = get_all_study_paths(Config.TRAIN_IMAGES_DIR, cache_key="train")

    def get_slice_count(uid):
        return len(paths_dict.get(uid, []))

    study_error["slice_count"] = study_error["StudyInstanceUID"].apply(get_slice_count)

    # Calculate Correlation
    if len(study_error) > 1:
        corr = study_error["error"].corr(study_error["slice_count"])
        print(f"Correlation between Mean Abs Error and Slice Count: {corr}")
    else:
        print("Not enough validation samples for correlation analysis.")

    # 6. Submission
    THRESHOLD = 0.15364714496434773

    if best_metric < THRESHOLD:
        # Load Test Metadata
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        # Test Dataset & Loader
        test_dataset = CervicalSpineDataset(
            test_df, mode="test", transform=get_transforms("test")
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        all_probs = []
        all_uids = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["images"].to(device)
                uids = batch["row_id"]

                # Forward
                outputs = model(images)
                logits = outputs["logits"]
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.append(probs)
                all_uids.extend(uids)

        all_probs = np.concatenate(all_probs, axis=0)

        # Format Submission
        class_names = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]
        submission_rows = []

        for i, uid in enumerate(all_uids):
            for class_idx, class_name in enumerate(class_names):
                row_id = f"{uid}_{class_name}"
                prob = all_probs[i, class_idx]
                submission_rows.append({"row_id": row_id, "fractured": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
