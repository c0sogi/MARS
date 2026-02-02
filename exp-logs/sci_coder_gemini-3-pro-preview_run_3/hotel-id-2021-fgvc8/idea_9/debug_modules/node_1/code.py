import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, mean_average_precision
from library.dataset import HotelDataset, get_transforms, get_label_mapping
from library.model import HotelModel, GeM, SubCenterArcFace
from library.engine import train_one_epoch, evaluate, inference


def run_demo():
    # -------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------
    print("Initializing Configuration...")
    config = Config(debug=True)

    # Override paths for the demo to keep things clean
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)
    config.working_dir = demo_dir
    config.submission_path = os.path.join(demo_dir, "demo_submission.csv")

    # Set seed
    seed_everything(config.seed)
    print(f"Working Directory: {config.working_dir}")
    print(f"Device: {config.device}")

    # -------------------------------------------------------------------
    # 2. Logic Verification (Unit Tests)
    # -------------------------------------------------------------------
    print("\n--- Verifying Logic Components ---")

    # A. Verify MAP@5 Metric
    print("Verifying MAP@5 metric...")
    # Scenario:
    # Target: Class 1
    # Preds: [1, 2, 3, 4, 5] -> Rank 1 -> Score 1.0
    # Target: Class 2
    # Preds: [1, 3, 2, 4, 5] -> Rank 3 -> Score 1/3
    dummy_targets = torch.tensor([1, 2])
    dummy_preds = torch.tensor([[1, 2, 3, 4, 5], [1, 3, 2, 4, 5]])
    expected_map = (1.0 + 1.0 / 3.0) / 2.0
    calculated_map = mean_average_precision(dummy_preds, dummy_targets, k=5)

    assert np.isclose(
        calculated_map, expected_map
    ), f"MAP@5 verification failed. Expected {expected_map}, got {calculated_map}"
    print("MAP@5 verification passed.")

    # B. Verify GeM Pooling
    print("Verifying GeM Pooling...")
    gem = GeM(p=3)
    # Input: (Batch=2, Channels=64, H=10, W=10)
    dummy_features = torch.randn(2, 64, 10, 10)
    pooled = gem(dummy_features)
    # Output should be (Batch=2, Channels=64, 1, 1)
    assert pooled.shape == (
        2,
        64,
        1,
        1,
    ), f"GeM output shape mismatch. Expected (2, 64, 1, 1), got {pooled.shape}"
    print("GeM Pooling verification passed.")

    # C. Verify SubCenterArcFace Head
    print("Verifying SubCenterArcFace Head...")
    batch_size = 4
    emb_dim = 128
    n_classes = 10
    k = 3
    head = SubCenterArcFace(in_features=emb_dim, out_features=n_classes, k=k)

    dummy_emb = torch.randn(batch_size, emb_dim)
    dummy_lbl = torch.randint(0, n_classes, (batch_size,))

    # Forward pass (Training mode)
    output = head(dummy_emb, dummy_lbl)
    assert output.shape == (
        batch_size,
        n_classes,
    ), f"Head output shape mismatch. Expected ({batch_size}, {n_classes}), got {output.shape}"

    # Forward pass (Inference mode)
    output_inf = head(dummy_emb, labels=None)
    assert output_inf.shape == (
        batch_size,
        n_classes,
    ), f"Head inference output shape mismatch. Expected ({batch_size}, {n_classes}), got {output_inf.shape}"
    print("SubCenterArcFace verification passed.")

    # -------------------------------------------------------------------
    # 3. Data Loading & Preparation
    # -------------------------------------------------------------------
    print("\n--- Preparing Data ---")

    # Load metadata
    train_df = pd.read_csv(config.train_csv_path)
    val_df = pd.read_csv(config.val_csv_path)
    test_df = pd.read_csv(config.test_csv_path)

    # OPTIMIZATION: Subsample data for speed
    # We select a small subset of classes to ensure the model can actually run
    # without needing to load thousands of images.
    subset_size = 32
    train_subset = train_df.iloc[:subset_size].copy()
    val_subset = val_df.iloc[:subset_size].copy()
    test_subset = test_df.iloc[:subset_size].copy()

    print(
        f"Subsampled Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # Generate Label Mapping
    # Note: We must ensure the mapping covers the classes in our subset
    # Cite debug_lesson_3: Derive Global Mappings from Full Datasets Before Subsetting
    class_to_idx, idx_to_class = get_label_mapping(
        train_df, config.working_dir, load_cached_data=False
    )
    n_classes_demo = len(class_to_idx)
    print(f"Number of classes in demo subset: {n_classes_demo}")

    # Create Datasets
    train_dataset = HotelDataset(
        train_subset,
        config.image_root_dir,
        transform=get_transforms(config.stage1_resolution, mode="train"),
        mode="train",
        class_to_idx=class_to_idx,
    )

    val_dataset = HotelDataset(
        val_subset,
        config.image_root_dir,
        transform=get_transforms(config.stage1_resolution, mode="val"),
        mode="val",
        class_to_idx=class_to_idx,
    )

    test_dataset = HotelDataset(
        test_subset,
        config.image_root_dir,
        transform=get_transforms(config.inference_resolution, mode="test"),
        mode="test",
    )

    # Create DataLoaders
    # Using a small batch size for the demo
    demo_batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=demo_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=demo_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=demo_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # -------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    model = HotelModel(
        backbone_name=config.backbone_name,
        n_classes=n_classes_demo,  # Use the subset class count
        embedding_dim=config.embedding_dim,
        pretrained=True,
        use_gem_pooling=config.use_gem_pooling,
        use_bn_neck=config.use_bn_neck,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        sub_centers_k=config.sub_centers_k,
    )
    model.to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.stage1_lr, weight_decay=config.weight_decay
    )

    # -------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------
    print("\n--- Starting Training Demo ---")
    epoch = 1
    avg_loss = train_one_epoch(model, optimizer, train_loader, config.device, epoch)

    assert not np.isnan(avg_loss), "Training loss returned NaN!"
    print(f"Training demo complete. Avg Loss: {avg_loss:.4f}")

    # -------------------------------------------------------------------
    # 6. Evaluation Demonstration
    # -------------------------------------------------------------------
    print("\n--- Starting Evaluation Demo ---")
    val_loss, val_map = evaluate(model, val_loader, config.device)

    assert not np.isnan(val_loss), "Validation loss returned NaN!"
    assert 0.0 <= val_map <= 1.0, f"Validation MAP score out of range: {val_map}"
    print(f"Evaluation demo complete. Val Loss: {val_loss:.4f}, MAP@5: {val_map:.4f}")

    # -------------------------------------------------------------------
    # 7. Inference Demonstration
    # -------------------------------------------------------------------
    print("\n--- Starting Inference Demo ---")

    # For inference, we need the full class mapping (or at least the one used during training)
    # Here we use the demo mapping
    inference(
        model,
        test_loader,
        config.device,
        idx_to_class,
        config.submission_path,
        use_tta=False,  # Disable TTA for speed in demo
    )

    # Verify submission file
    if os.path.exists(config.submission_path):
        sub_df = pd.read_csv(config.submission_path)
        print(f"Submission file created at {config.submission_path}")
        print(f"Submission shape: {sub_df.shape}")
        print("Head:")
        print(sub_df.head())

        assert sub_df.shape[0] == len(
            test_subset
        ), f"Submission rows {sub_df.shape[0]} != Test subset size {len(test_subset)}"
        assert (
            "image" in sub_df.columns and "hotel_id" in sub_df.columns
        ), "Submission columns missing."
    else:
        raise FileNotFoundError("Submission file was not created.")

    # Save the demo model
    demo_model_path = os.path.join(demo_dir, "demo_model.pth")
    torch.save(model.state_dict(), demo_model_path)
    print(f"Demo model saved to {demo_model_path}")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
