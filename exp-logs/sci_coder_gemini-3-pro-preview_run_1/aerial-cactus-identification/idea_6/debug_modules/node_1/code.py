import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import load_data_to_memory, CactusDataset, get_transforms
from library.model import RepVGGDeepSup
from library.engine import train_model, predict_tta, set_seed


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print("Setting up configuration for demo run...")
    set_seed(Config.SEED)

    # Override Config for a fast demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32  # Smaller batch size for the small subset
    Config.WORKING_DIR = "./working/demo_run"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.print_summary()

    # 2. Data Loading & Subsampling
    print("\n[Data] Loading and subsampling data...")

    # Load Training Data
    train_imgs, train_labels = load_data_to_memory(
        Config.TRAIN_META_PATH,
        os.path.join(Config.WORKING_DIR, "cache_train_imgs.npy"),
        os.path.join(Config.WORKING_DIR, "cache_train_labels.npy"),
        load_cached_data=False,  # Force reload for demo to ensure logic works
    )

    # Load Validation Data
    val_imgs, val_labels = load_data_to_memory(
        Config.VAL_META_PATH,
        os.path.join(Config.WORKING_DIR, "cache_val_imgs.npy"),
        os.path.join(Config.WORKING_DIR, "cache_val_labels.npy"),
        load_cached_data=False,
    )

    # Subsample for speed (use only 500 train, 100 val)
    subset_train_size = 500
    subset_val_size = 100

    print(f"  Subsampling Train: {len(train_imgs)} -> {subset_train_size}")
    train_imgs = train_imgs[:subset_train_size]
    train_labels = train_labels[:subset_train_size]

    print(f"  Subsampling Val:   {len(val_imgs)} -> {subset_val_size}")
    val_imgs = val_imgs[:subset_val_size]
    val_labels = val_labels[:subset_val_size]

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_labels, transform=get_transforms(split="train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_labels, transform=get_transforms(split="val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 3. Model Logic Verification
    print("\n[Model] Verifying RepVGGDeepSup logic...")
    device = Config.DEVICE
    model_verify = RepVGGDeepSup(num_classes=1, deploy=False).to(device)

    # Create dummy input
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    # Check Training Mode Output (Deep Supervision: Main + Aux)
    model_verify.train()
    out = model_verify(dummy_input)
    assert (
        isinstance(out, tuple) and len(out) == 2
    ), "Model in train mode should return (main, aux)"
    assert out[0].shape == (4, 1), f"Main output shape mismatch: {out[0].shape}"
    assert out[1].shape == (4, 1), f"Aux output shape mismatch: {out[1].shape}"
    print("  Training mode output verified (Main + Aux).")

    # Check Deploy Mode Switching
    model_verify.eval()
    model_verify.switch_to_deploy()
    assert model_verify.deploy is True, "Model deploy flag should be True after switch"
    assert not hasattr(
        model_verify, "aux_head"
    ), "Aux head should be removed in deploy mode"

    out_deploy = model_verify(dummy_input)
    assert isinstance(
        out_deploy, torch.Tensor
    ), "Deploy model should return a single tensor"
    assert out_deploy.shape == (
        4,
        1,
    ), f"Deploy output shape mismatch: {out_deploy.shape}"
    print("  Deploy mode switch and output verified.")

    del model_verify, dummy_input, out, out_deploy
    torch.cuda.empty_cache()

    # 4. Training Loop
    print("\n[Training] Starting training loop...")

    # Instantiate fresh model for training
    model = RepVGGDeepSup(num_classes=1, deploy=False).to(device)

    best_auc = train_model(model, train_loader, val_loader, device)

    print(f"  Training finished. Best AUC on subset: {best_auc:.4f}")
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."

    # 5. Inference
    print("\n[Inference] Running prediction on Test set...")

    # Load Test Data
    test_imgs, test_ids = load_data_to_memory(
        Config.TEST_META_PATH,
        os.path.join(Config.WORKING_DIR, "cache_test_imgs.npy"),
        os.path.join(
            Config.WORKING_DIR, "cache_test_labels.npy"
        ),  # Not used but path required
        os.path.join(Config.WORKING_DIR, "cache_test_ids.npy"),
        load_cached_data=False,
        is_test=True,
    )

    # Subsample Test Data
    subset_test_size = 100
    print(f"  Subsampling Test: {len(test_imgs)} -> {subset_test_size}")
    test_imgs = test_imgs[:subset_test_size]
    test_ids = test_ids[:subset_test_size]

    test_dataset = CactusDataset(
        test_imgs, labels=None, ids=test_ids, transform=get_transforms(split="test")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load Best Model
    # Note: We must instantiate a non-deploy model first, load weights, then switch to deploy
    # because the checkpoint was saved during training (non-deploy architecture).
    inference_model = RepVGGDeepSup(num_classes=1, deploy=False).to(device)
    inference_model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=device)
    )

    # Predict (TTA handles switch_to_deploy internally)
    predictions_dict = predict_tta(inference_model, test_loader, device)

    # 6. Submission Generation
    print("\n[Submission] Generating submission file...")

    # Create DataFrame
    sub_df = pd.DataFrame(
        {
            "id": list(predictions_dict.keys()),
            "has_cactus": list(predictions_dict.values()),
        }
    )

    # Verify values
    assert sub_df["has_cactus"].min() >= 0.0, "Probabilities must be >= 0"
    assert sub_df["has_cactus"].max() <= 1.0, "Probabilities must be <= 1"
    assert len(sub_df) == subset_test_size, "Submission size mismatch"

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"  Submission saved to {Config.SUBMISSION_PATH}")
    print("\nDemo run completed successfully.")


if __name__ == "__main__":
    run_demo()
