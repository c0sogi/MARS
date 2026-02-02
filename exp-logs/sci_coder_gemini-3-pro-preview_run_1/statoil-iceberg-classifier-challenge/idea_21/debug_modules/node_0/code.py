import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import SEED, WORK_DIR
from library.utils import set_seed, save_checkpoint
from library.dataset import load_data_splits, IcebergDataset, get_transforms
from library.network import IcebergResNet
from library.engine import (
    get_optimizer_scheduler,
    train_one_epoch,
    evaluate,
    get_swa_model,
    update_swa_bn,
    generate_submission,
)
from library.pseudo_labeling import (
    generate_ensemble_stats,
    filter_pseudo_labels,
    extract_pseudo_dataset,
)


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Initialization
    # -------------------------------------------------------------------------
    print("--- Starting Demo Execution ---")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define demo-specific directories to avoid overwriting production files
    demo_dir = os.path.join(os.path.dirname(WORK_DIR), "demo_execution")
    checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    submission_dir = os.path.join(demo_dir, "submission")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading & Subsetting
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and subsetting data...")
    # Load full data (cached or raw)
    train_data_full, val_data_full, test_data_full = load_data_splits(
        load_cached_data=True
    )

    # Create small subsets for speed
    N_TRAIN = 64
    N_VAL = 32
    N_TEST = 32

    def subset_dict(data_dict, n):
        return {k: v[:n] if v is not None else None for k, v in data_dict.items()}

    train_sub = subset_dict(train_data_full, N_TRAIN)
    val_sub = subset_dict(val_data_full, N_VAL)
    test_sub = subset_dict(test_data_full, N_TEST)

    # Initialize Datasets
    train_ds = IcebergDataset(
        train_sub["images"],
        train_sub["angles"],
        train_sub["labels"],
        train_sub["ids"],
        transform=get_transforms("train"),
    )
    val_ds = IcebergDataset(
        val_sub["images"],
        val_sub["angles"],
        val_sub["labels"],
        val_sub["ids"],
        transform=get_transforms("val"),
    )
    test_ds = IcebergDataset(
        test_sub["images"],
        test_sub["angles"],
        test_sub["labels"],
        test_sub["ids"],
        transform=get_transforms("test"),
    )

    # Initialize DataLoaders
    # Using batch_size=16 for the demo
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    print(
        f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}"
    )

    # -------------------------------------------------------------------------
    # 3. Model Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 2] Initializing model and running training loop...")
    model = IcebergResNet().to(device)
    optimizer, scheduler = get_optimizer_scheduler(model)

    # Run for 2 epochs
    for epoch in range(1, 3):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, preds, targets = evaluate(model, val_loader, device)

        # Validation
        if np.isnan(train_loss):
            raise ValueError("Training loss is NaN.")
        if len(preds) != N_VAL:
            raise AssertionError(
                f"Prediction count mismatch. Expected {N_VAL}, got {len(preds)}"
            )

    print("Training loop completed successfully.")

    # Save Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    save_checkpoint(
        {
            "epoch": 2,
            "state_dict": model.state_dict(),
            "best_metric": val_loss,
        },
        is_best=True,
        checkpoint_dir=checkpoint_dir,
        filename="checkpoint.pth",
    )

    # Save a second copy to simulate an ensemble later
    swa_checkpoint_path = os.path.join(checkpoint_dir, "swa_model.pth")
    torch.save({"state_dict": model.state_dict()}, swa_checkpoint_path)

    # -------------------------------------------------------------------------
    # 4. SWA Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 3] Demonstrating SWA setup...")
    swa_model = get_swa_model(model)
    # Update BN stats using the training loader
    update_swa_bn(train_loader, swa_model, device)

    # Verify SWA model inference
    swa_model.eval()
    with torch.no_grad():
        imgs, angs, _, _ = next(iter(val_loader))
        out = swa_model(imgs.to(device), angs.to(device))
        if out.shape != (imgs.shape[0], 1):
            raise AssertionError(
                f"SWA output shape incorrect. Expected {(imgs.shape[0], 1)}, got {out.shape}"
            )
    print("SWA BN update and inference verified.")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating submission...")
    sub_path = os.path.join(submission_dir, "submission.csv")
    generate_submission(model, test_loader, device, sub_path)

    # Verify submission file
    if not os.path.exists(sub_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(sub_path)
    if len(df_sub) != N_TEST:
        raise AssertionError(
            f"Submission rows mismatch. Expected {N_TEST}, got {len(df_sub)}"
        )
    if list(df_sub.columns) != ["id", "is_iceberg"]:
        raise AssertionError("Submission columns mismatch.")

    print(f"Submission saved to {sub_path}")

    # -------------------------------------------------------------------------
    # 6. Pseudo-Labeling Workflow
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Pseudo-Labeling workflow...")

    # We use the two checkpoints saved earlier to simulate an ensemble
    model_paths = [
        os.path.join(checkpoint_dir, "checkpoint.pth"),
        os.path.join(checkpoint_dir, "swa_model.pth"),
    ]

    # Generate stats (Mean/Std)
    stats_df = generate_ensemble_stats(
        model_paths,
        test_loader,
        device,
        load_cached_data=False,  # Force computation
        cache_dir=demo_dir,
    )

    # Filter Pseudo-Labels
    # Using very loose thresholds just to ensure we select *some* data for the demo
    pseudo_labels_df = filter_pseudo_labels(
        stats_df,
        conf_high=0.4,  # Lowered for demo
        conf_low=0.6,  # Raised for demo
        var_thresh=1.0,  # High variance allowed for demo
    )

    # Extract Data
    p_images, p_angles, p_labels, p_ids = extract_pseudo_dataset(
        test_sub, pseudo_labels_df
    )

    # Verification
    if len(p_images) > 0:
        if p_images.shape[1:] != (75, 75, 2):
            raise AssertionError(f"Pseudo-image shape incorrect: {p_images.shape}")
        if len(p_labels) != len(p_images):
            raise AssertionError("Mismatch between pseudo-images and labels count.")
        print(f"Successfully extracted {len(p_images)} pseudo-labeled samples.")
    else:
        print(
            "No pseudo-labels selected (this is possible with random weights/small data)."
        )

    print("\n--- Demo Execution Completed Successfully ---")


if __name__ == "__main__":
    main()
