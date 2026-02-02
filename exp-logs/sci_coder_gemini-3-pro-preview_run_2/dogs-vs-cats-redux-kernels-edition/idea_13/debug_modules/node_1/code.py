import os
import sys
import torch
import pandas as pd
import numpy as np
import glob
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    calc_log_loss,
)
from library.dataset import DogCatDataset, get_transforms, MixupCutmixCollate
from library.models import get_model
from library.engine import train_one_epoch, validate, inference_fn
from library.soup import generate_soup_model
from library.stacking import (
    load_aggregated_predictions,
    fit_meta_learner,
    predict_meta_learner,
    create_submission,
)


def run_demo():
    print("=== Starting Library Validation Demo ===\n")

    # 1. Setup and Seeding
    print("--- Step 1: Initialization ---")
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # Define working directories for the demo
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    oof_dir = os.path.join(demo_dir, "oof")
    cache_dir = os.path.join(demo_dir, "cache")
    submission_dir = os.path.join(demo_dir, "submission")

    for d in [demo_dir, checkpoint_dir, oof_dir, cache_dir, submission_dir]:
        os.makedirs(d, exist_ok=True)

    # Override Config paths temporarily for the demo to point to our local demo folders
    # We do this by monkey-patching the Config class attributes or just passing paths where functions allow.
    # Since functions like load_aggregated_predictions use Config.OOF_DIR directly, we patch Config.
    original_oof_dir = Config.OOF_DIR
    original_cache_dir = Config.CACHE_DIR
    Config.OOF_DIR = oof_dir
    Config.CACHE_DIR = cache_dir

    # 2. Dataset and Dataloader Validation
    print("\n--- Step 2: Data Loading & Augmentation ---")

    # Load metadata
    train_df_full = pd.read_csv(Config.TRAIN_METADATA)
    val_df_full = pd.read_csv(Config.VAL_METADATA)
    test_df_full = pd.read_csv(Config.TEST_METADATA)

    # Create tiny subsets for speed
    train_subset = train_df_full.head(32).copy()  # 1 batch
    val_subset = val_df_full.head(16).copy()
    test_subset = test_df_full.head(16).copy()

    print(f"Train subset size: {len(train_subset)}")
    print(f"Val subset size: {len(val_subset)}")

    # Instantiate Datasets
    train_dataset = DogCatDataset(
        train_subset, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = DogCatDataset(
        val_subset, transforms=get_transforms("val"), mode="val"
    )
    test_dataset = DogCatDataset(
        test_subset, transforms=get_transforms("test"), mode="test"
    )

    # Test __getitem__
    img, label = train_dataset[0]
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    print("Dataset __getitem__ check passed.")

    # Instantiate DataLoaders
    # We use a small batch size for the demo
    batch_size = 8
    mixup_fn = MixupCutmixCollate(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mixup_fn,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Test DataLoader batch
    imgs, targets = next(iter(train_loader))
    assert imgs.shape == (
        batch_size,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Batch image shape mismatch"
    assert targets.shape == (batch_size,), "Batch target shape mismatch"
    print("DataLoader batch check passed.")

    # 3. Model Creation & Training
    print("\n--- Step 3: Model Training (ResNet18) ---")

    # Use a lightweight model for the demo instead of the heavy ones in Config
    model_name = "resnet18"
    model = get_model(model_name, pretrained=True, num_classes=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train 1 epoch
    print("Training for 1 epoch...")
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch=1)
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    print("Validating...")
    val_loss, val_metric = validate(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")

    # Save Checkpoint 1
    ckpt_path_1 = os.path.join(checkpoint_dir, "resnet18_fold_0.pth")
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path_1,
    )
    assert os.path.exists(ckpt_path_1), "Checkpoint 1 not saved"

    # Simulate a second checkpoint (slightly different weights)
    # We just run one more step or modify weights manually to ensure soup is different
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)

    ckpt_path_2 = os.path.join(checkpoint_dir, "resnet18_fold_1.pth")
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
        },
        ckpt_path_2,
    )
    print("Checkpoints saved.")

    # 4. Model Soup
    print("\n--- Step 4: Model Soup ---")

    # Create a fresh model instance
    soup_model = get_model(model_name, pretrained=False, num_classes=1)
    soup_model.to(device)

    # Generate soup
    soup_model = generate_soup_model(
        soup_model, [ckpt_path_1, ckpt_path_2], device="cpu"
    )
    soup_model.to(device)

    # Save soup model
    soup_path = os.path.join(checkpoint_dir, "best_resnet18_soup.pth")
    save_checkpoint(soup_model.state_dict(), soup_path)
    print("Model soup generated and saved.")

    # 5. Inference
    print("\n--- Step 5: Inference (TTA) ---")
    ids, preds = inference_fn(soup_model, test_loader, device)

    assert len(ids) == len(test_subset), "Inference ID count mismatch"
    assert len(preds) == len(test_subset), "Inference prediction count mismatch"
    assert all(0.0 <= p <= 1.0 for p in preds), "Predictions out of probability range"
    print(f"Generated {len(preds)} predictions.")

    # 6. Stacking / Meta-Learner
    print("\n--- Step 6: Stacking & Meta-Learner ---")

    # Generate dummy OOF and Test predictions for 2 models to demonstrate stacking
    # Model A
    model_a_oof = pd.DataFrame(
        {
            "id": train_subset["filepath"].apply(
                lambda x: hash(x) % 10000
            ),  # Dummy IDs
            "target": train_subset["label"],
            "pred": np.random.uniform(0, 1, len(train_subset)),
        }
    )
    # Ensure IDs are unique for merge logic
    model_a_oof["id"] = range(1, len(model_a_oof) + 1)

    model_a_test = pd.DataFrame(
        {"id": test_subset["id"], "pred": np.random.uniform(0, 1, len(test_subset))}
    )

    # Model B
    model_b_oof = model_a_oof.copy()
    model_b_oof["pred"] = np.random.uniform(0, 1, len(train_subset))

    model_b_test = model_a_test.copy()
    model_b_test["pred"] = np.random.uniform(0, 1, len(test_subset))

    # Save these to the OOF directory
    model_a_oof.to_csv(os.path.join(oof_dir, "model_a_oof.csv"), index=False)
    model_b_oof.to_csv(os.path.join(oof_dir, "model_b_oof.csv"), index=False)
    model_a_test.to_csv(os.path.join(oof_dir, "model_a_test.csv"), index=False)
    model_b_test.to_csv(os.path.join(oof_dir, "model_b_test.csv"), index=False)

    # Load Aggregated OOF
    print("Loading aggregated OOF...")
    # Force reload by ensuring cache doesn't exist or load_cached_data=False
    if os.path.exists(os.path.join(cache_dir, "meta_oof_data.parquet")):
        os.remove(os.path.join(cache_dir, "meta_oof_data.parquet"))

    agg_oof_df = load_aggregated_predictions(mode="oof", load_cached_data=False)
    assert (
        "model_a" in agg_oof_df.columns and "model_b" in agg_oof_df.columns
    ), "Columns missing in aggregated OOF"
    assert "target" in agg_oof_df.columns, "Target missing in aggregated OOF"

    # Load Aggregated Test
    print("Loading aggregated Test...")
    agg_test_df = load_aggregated_predictions(mode="test", load_cached_data=False)
    assert (
        "model_a" in agg_test_df.columns and "model_b" in agg_test_df.columns
    ), "Columns missing in aggregated Test"

    # Fit Meta Learner
    print("Fitting Meta-Learner...")
    meta_model, feature_cols = fit_meta_learner(agg_oof_df, target_col="target")

    # Predict
    print("Predicting with Meta-Learner...")
    final_preds = predict_meta_learner(meta_model, agg_test_df, feature_cols)

    # Create Submission
    sub_path = os.path.join(submission_dir, "submission.csv")
    create_submission(agg_test_df, final_preds, sub_path)

    assert os.path.exists(sub_path), "Submission file not created"

    # Verify submission content
    sub_df = pd.read_csv(sub_path)
    print("Submission head:")
    print(sub_df.head())
    assert sub_df.shape == (len(test_subset), 2), "Submission shape mismatch"

    # Restore Config
    Config.OOF_DIR = original_oof_dir
    Config.CACHE_DIR = original_cache_dir

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
