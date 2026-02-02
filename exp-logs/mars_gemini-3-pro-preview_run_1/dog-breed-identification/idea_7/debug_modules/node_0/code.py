import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MixupCutmix
from library.dataset import get_datasets, get_transforms, DogDataset
from library.model import get_model, set_backbone_trainable
from library.engine import (
    train_one_epoch,
    validate,
    predict,
    save_submission,
    update_swa_model,
    update_bn_statistics,
)


def run_demo():
    # 1. Setup and Configuration
    print("--- Setting up Demo Configuration ---")
    seed_everything(Config.SEED)

    # Define a temporary working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.OUTPUT_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override compute parameters for speed optimization
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.RESIZE_DIM = 64  # Small size for speed
    Config.CROP_SIZE = 64

    # 2. Create Subset Metadata
    # We create tiny subsets of the original metadata to ensure the code runs instantly
    print("--- Creating Data Subsets ---")

    # Load original metadata
    train_meta_orig = pd.read_csv(Config.TRAIN_METADATA)
    val_meta_orig = pd.read_csv(Config.VAL_METADATA)
    test_meta_orig = pd.read_csv(Config.TEST_METADATA)

    # Sample subsets (16 train, 8 val, 8 test)
    train_subset = train_meta_orig.head(16)
    val_subset = val_meta_orig.head(8)
    test_subset = test_meta_orig.head(8)

    # Save subsets to demo directory
    train_subset_path = os.path.join(demo_dir, "train_subset.csv")
    val_subset_path = os.path.join(demo_dir, "val_subset.csv")
    test_subset_path = os.path.join(demo_dir, "test_subset.csv")

    train_subset.to_csv(train_subset_path, index=False)
    val_subset.to_csv(val_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    # Point Config to these new files
    Config.TRAIN_METADATA = train_subset_path
    Config.VAL_METADATA = val_subset_path
    Config.TEST_METADATA = test_subset_path

    # 3. Data Loading
    print("--- Demonstrating Data Loading ---")
    # get_datasets handles loading, caching, and label mapping
    # We set load_cached_data=False to force processing of our new subsets
    (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, _), label_map = (
        get_datasets(load_cached_data=False)
    )

    print(f"Train Images Shape: {train_imgs.shape}")
    print(f"Train Labels Shape: {train_lbls.shape}")
    print(f"Label Map Size: {len(label_map)}")

    # Verify data integrity
    assert len(train_imgs) == 16
    assert len(val_imgs) == 8
    assert len(test_imgs) == 8
    assert train_imgs.shape[1:] == (Config.CROP_SIZE, Config.CROP_SIZE, 3)

    # 4. Dataset & DataLoader
    print("--- Demonstrating Dataset & DataLoader ---")
    train_transforms = get_transforms(mode="train", input_size=Config.CROP_SIZE)
    val_transforms = get_transforms(mode="val", input_size=Config.CROP_SIZE)

    train_dataset = DogDataset(train_imgs, train_lbls, transforms=train_transforms)
    val_dataset = DogDataset(val_imgs, val_lbls, transforms=val_transforms)
    test_dataset = DogDataset(test_imgs, None, transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify batch structure
    batch_imgs, batch_lbls = next(iter(train_loader))
    assert batch_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    )
    assert batch_lbls.shape == (Config.BATCH_SIZE,)

    # 5. Model Instantiation
    print("--- Demonstrating Model Creation ---")
    model_config = Config.MODEL_CONFIGS["convnext_base"]
    # Disable pretrained weights download for speed and to ensure offline execution
    model_config["pretrained"] = False
    # Ensure input size matches our tiny crop
    model_config["input_size"] = Config.CROP_SIZE

    model = get_model(model_config)
    model.to(Config.DEVICE)

    # Verify forward pass
    with torch.no_grad():
        dummy_out = model(batch_imgs.to(Config.DEVICE))
    assert dummy_out.shape == (Config.BATCH_SIZE, 120)  # 120 classes

    # Demonstrate freezing backbone
    set_backbone_trainable(model, trainable=False)
    # Check if backbone is frozen (ConvNeXt typically has 'stem' or 'stages')
    if hasattr(model, "stem"):
        assert model.stem[0].weight.requires_grad == False
        print("Backbone frozen successfully.")

    set_backbone_trainable(model, trainable=True)
    if hasattr(model, "stem"):
        assert model.stem[0].weight.requires_grad == True
        print("Backbone unfrozen successfully.")

    # 6. Training Loop Component (Mixup)
    print("--- Demonstrating Mixup/CutMix ---")
    # Force mix_prob=1.0 to ensure mixup logic is executed and tested
    mixup_fn = MixupCutmix(
        mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=1.0, num_classes=120
    )
    mixed_imgs, mixed_lbls = mixup_fn(batch_imgs, batch_lbls)

    assert mixed_imgs.shape == batch_imgs.shape
    # Labels should be one-hot encoded (Batch, Num_Classes) after mixup
    assert mixed_lbls.shape == (Config.BATCH_SIZE, 120)

    # 7. Engine: Train One Epoch
    print("--- Demonstrating Training Step ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run 1 epoch
    train_loss = train_one_epoch(
        model, optimizer, train_loader, Config.DEVICE, epoch=1, mixup_fn=mixup_fn
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss)

    # 8. Engine: Validation
    print("--- Demonstrating Validation ---")
    val_loss = validate(model, val_loader, Config.DEVICE)
    print(f"Validation Log Loss: {val_loss:.4f}")
    assert not np.isnan(val_loss)

    # 9. Engine: Prediction & TTA
    print("--- Demonstrating Prediction ---")
    # Use TTA (Test Time Augmentation)
    preds = predict(model, test_loader, Config.DEVICE, use_tta=True)
    assert preds.shape == (len(test_imgs), 120)
    # Check probability validity
    assert np.all((preds >= 0) & (preds <= 1))

    # 10. SWA Utils
    print("--- Demonstrating SWA Utilities ---")
    # Create a wrapper model for SWA
    swa_model = torch.optim.swa_utils.AveragedModel(model)

    # Update SWA parameters with current model
    update_swa_model(swa_model, model)

    # Update BN statistics using the training loader
    update_bn_statistics(swa_model, train_loader, Config.DEVICE)
    print("SWA Model updated.")

    # 11. Submission
    print("--- Demonstrating Submission Generation ---")
    save_submission(preds, Config.TEST_METADATA, label_map, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH)
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    # Shape should be (N_test, 1 id col + 120 breed cols)
    assert sub_df.shape == (len(test_imgs), 121)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
