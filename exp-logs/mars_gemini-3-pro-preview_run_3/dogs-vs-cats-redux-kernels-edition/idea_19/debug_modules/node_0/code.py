import os
import shutil
import torch
import pandas as pd
import numpy as np
import library.config as config
import library.data as data
import library.models as models
import library.engine as engine
import library.inference as inference
import library.utils as utils


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Setup & Reproducibility
    # -------------------------------------------------------------------------
    print("\n[1] Setting up environment and seeds...")
    config.seed_everything(config.SEED)

    # Define temporary paths for mini-datasets
    mini_train_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test.csv")
    folds_cache_path = os.path.join(config.WORKING_DIR, "folds.parquet")

    # Clean up previous cache to ensure we use our mini datasets
    if os.path.exists(folds_cache_path):
        os.remove(folds_cache_path)

    # -------------------------------------------------------------------------
    # 2. Create Mini Datasets (for speed)
    # -------------------------------------------------------------------------
    print("\n[2] Creating mini-datasets from existing metadata...")

    # Load original metadata to get valid file paths
    orig_train_df = pd.read_csv(config.TRAIN_META_PATH)
    orig_val_df = pd.read_csv(config.VAL_META_PATH)
    orig_test_df = pd.read_csv(config.TEST_META_PATH)

    # Sample a small subset (e.g., 20 train, 10 val, 10 test)
    # We ensure we have both classes in train/val for stability
    mini_train = pd.concat(
        [
            orig_train_df[orig_train_df["label"] == 0].head(10),
            orig_train_df[orig_train_df["label"] == 1].head(10),
        ]
    ).reset_index(drop=True)

    mini_val = pd.concat(
        [
            orig_val_df[orig_val_df["label"] == 0].head(5),
            orig_val_df[orig_val_df["label"] == 1].head(5),
        ]
    ).reset_index(drop=True)

    mini_test = orig_test_df.head(10).reset_index(drop=True)

    # Save mini datasets
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"    Mini Train: {len(mini_train)} rows")
    print(f"    Mini Val:   {len(mini_val)} rows")
    print(f"    Mini Test:  {len(mini_test)} rows")

    # -------------------------------------------------------------------------
    # 3. Monkey Patch Library Paths
    # -------------------------------------------------------------------------
    print("\n[3] Patching library paths to use mini-datasets...")
    # We must patch the variables in the `library.data` module namespace
    data.TRAIN_META_PATH = mini_train_path
    data.VAL_META_PATH = mini_val_path
    data.TEST_META_PATH = mini_test_path

    # Also patch config just in case other modules reference it (though they shouldn't)
    config.TRAIN_META_PATH = mini_train_path
    config.VAL_META_PATH = mini_val_path
    config.TEST_META_PATH = mini_test_path

    # -------------------------------------------------------------------------
    # 4. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Data Loading...")

    # Create a custom config for the demo (faster training)
    demo_cfg = config.ModelConfig(
        model_name="resnet18",  # Smaller model for demo
        weights="resnet18.a1_in1k",
        img_size=224,
        epochs=1,  # Only 1 epoch
        batch_size=4,  # Small batch size
        learning_rate=1e-4,
        min_lr=1e-6,
        weight_decay=1e-4,
    )

    # Test Dataset Class directly
    dataset = data.DogCatDataset(
        mini_train,
        root_dir=config.INPUT_DIR,
        transform=data.get_transforms(demo_cfg, is_train=True),
    )
    sample_img, sample_label = dataset[0]

    # Assertions
    assert isinstance(sample_img, torch.Tensor), "Dataset should return a Tensor image"
    assert sample_img.shape == (
        3,
        224,
        224,
    ), f"Image shape mismatch. Expected (3, 224, 224), got {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a Tensor"
    print("    DogCatDataset verification passed.")

    # Test Data Loaders generation (via get_fold_loaders)
    # Note: This will generate folds.parquet from our mini-datasets
    train_loader, val_loader = data.get_fold_loaders(
        fold_idx=0, cfg=demo_cfg, load_cached_data=False
    )

    batch_imgs, batch_labels = next(iter(train_loader))
    assert batch_imgs.shape[0] == demo_cfg.batch_size, "Batch size mismatch in loader"
    print("    DataLoader verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Model Creation
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model Creation...")
    model = models.create_model(demo_cfg, pretrained=True)
    model = model.to(config.DEVICE)

    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, demo_cfg.img_size, demo_cfg.img_size).to(
            config.DEVICE
        )
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("    Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 6. Verify Training Loop (Engine)
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (Fold 0)...")

    # We use run_fold which orchestrates training.
    # It uses the patched data paths and our demo_cfg.
    best_loss = engine.run_fold(fold_idx=0, cfg=demo_cfg)

    assert isinstance(best_loss, float), "run_fold should return a float loss"
    print(f"    Training completed. Best Val Loss: {best_loss:.4f}")

    # Verify checkpoint existence
    checkpoint_path = os.path.join(
        config.WORKING_DIR, f"{demo_cfg.model_name}_fold_0.pth"
    )
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    print("    Checkpoint verification passed.")

    # -------------------------------------------------------------------------
    # 7. Verify Inference
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference on Test Set...")

    # Load the best model
    checkpoint = utils.load_checkpoint(
        f"{demo_cfg.model_name}_fold_0.pth", model, device=config.DEVICE
    )

    # Get test loader
    test_loader = data.get_test_loader(demo_cfg)

    # Predict
    probs, ids = inference.predict(model, test_loader, device=config.DEVICE)

    # Assertions
    assert len(probs) == len(
        mini_test
    ), f"Prediction count mismatch. Expected {len(mini_test)}, got {len(probs)}"
    assert len(ids) == len(mini_test), "ID count mismatch"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"
    print("    Inference verification passed.")

    # -------------------------------------------------------------------------
    # 8. Verify Utilities (Metric & Submission)
    # -------------------------------------------------------------------------
    print("\n[8] Verifying Utilities...")

    # Metric
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]
    loss = utils.compute_log_loss(y_true, y_pred)
    assert loss < 0.3, "Log loss calculation seems incorrect for good predictions"
    print(f"    Log Loss check: {loss:.4f}")

    # Submission
    sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    utils.save_submission(ids, probs, sub_path)
    assert os.path.exists(sub_path), "Submission file not created"

    # Check submission content
    sub_df = pd.read_csv(sub_path)
    assert list(sub_df.columns) == ["id", "label"], "Submission columns mismatch"
    assert len(sub_df) == len(mini_test), "Submission row count mismatch"
    print("    Submission file verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
