import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, collate_fn
from library.transforms import get_transforms
from library.dataset import CovidDataset
from library.model import SwinCascadeRCNN
from library.engine import run


def main():
    print("=== Starting Demonstration of Swin Cascade R-CNN Pipeline ===\n")

    # 1. Configuration Setup
    # We override the working directory to separate this demo from main experiments
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Initialize Config in Debug mode
    # Debug mode reduces dataset size to 100 and epochs to 2 (we override to 1 for speed)
    Config.setup(debug=True, epochs=1, batch_size=2)

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"\n[Setup] Running on device: {device}")
    print(f"[Setup] Working Directory: {Config.WORKING_DIR}")

    # 2. Verify Data Transformations
    print("\n=== 2. Verifying Augmentations/Transforms ===")
    try:
        transforms = get_transforms("train")
        # Create a dummy image (H, W, 3)
        dummy_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        # Dummy bounding box [xmin, ymin, xmax, ymax]
        dummy_bboxes = [[100, 100, 200, 200]]
        dummy_labels = [1]

        augmented = transforms(
            image=dummy_image, bboxes=dummy_bboxes, class_labels=dummy_labels
        )

        # Assertions
        assert "image" in augmented, "Transform output missing 'image' key"
        assert isinstance(
            augmented["image"], torch.Tensor
        ), "Augmented image is not a Tensor"
        assert augmented["image"].shape == (
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Incorrect image shape: {augmented['image'].shape}"
        assert len(augmented["bboxes"]) == 1, "Bounding box lost during augmentation"

        print("[Check] Transforms applied successfully. Output shape correct.")
    except Exception as e:
        print(f"[Error] Transform verification failed: {e}")
        raise e

    # 3. Verify Dataset Loading & Caching
    print("\n=== 3. Verifying Dataset & Caching ===")
    try:
        # Load metadata
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        # Use a tiny subset (5 images) for this specific check to be instant
        df_subset = df_train.head(5).copy()

        print(f"[Dataset] Initializing dataset with {len(df_subset)} samples...")
        dataset = CovidDataset(
            df_subset,
            transforms=get_transforms("train"),
            split="train",
            load_cached_data=True,
        )

        # Fetch one item
        img, target, img_id = dataset[0]

        # Assertions
        assert isinstance(img, torch.Tensor), "Dataset returned non-tensor image"
        assert img.shape == (
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), "Dataset image shape mismatch"
        assert isinstance(target, dict), "Target is not a dictionary"
        required_keys = ["boxes", "labels", "study_label", "image_id", "orig_size"]
        for k in required_keys:
            assert k in target, f"Target missing key: {k}"

        print(f"[Check] Successfully loaded image {img_id}")
        print(f"[Check] Target keys verified: {list(target.keys())}")
    except Exception as e:
        print(f"[Error] Dataset verification failed: {e}")
        raise e

    # 4. Verify Model Architecture
    print("\n=== 4. Verifying Model Architecture ===")
    try:
        print("[Model] Instantiating SwinCascadeRCNN...")
        model = SwinCascadeRCNN()
        model.to(device)

        # Prepare a dummy batch from the dataset item retrieved above
        # We need to collate it to simulate a DataLoader batch
        batch = [(img, target, img_id)]
        images_batch, targets_batch, ids_batch = collate_fn(batch)

        images_batch = images_batch.to(device)
        targets_batch = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

        # A. Training Forward Pass
        print("[Model] Running Training Forward Pass...")
        model.train()
        loss_dict = model(images_batch, targets_batch)

        # Assertions for Loss
        assert isinstance(
            loss_dict, dict
        ), "Model output in train mode should be a dict"
        print(f"[Check] Loss keys: {list(loss_dict.keys())}")
        assert "loss_study" in loss_dict, "Missing study loss"
        assert "loss_objectness" in loss_dict, "Missing RPN loss"

        # B. Inference Forward Pass
        print("[Model] Running Inference Forward Pass...")
        model.eval()
        with torch.no_grad():
            predictions = model(images_batch)

        # Assertions for Predictions
        assert isinstance(
            predictions, list
        ), "Model output in eval mode should be a list"
        assert len(predictions) == 1, "Prediction batch size mismatch"
        pred = predictions[0]
        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred
        assert "study_probs" in pred

        print("[Check] Model forward passes successful.")
    except Exception as e:
        print(f"[Error] Model verification failed: {e}")
        raise e

    # 5. Verify Full Training Loop (Engine)
    print("\n=== 5. Verifying Full Training Loop (Engine) ===")
    print("Running a short training cycle (1 epoch, debug sample size)...")
    try:
        # We call the `run` function from engine.py
        # This will reload the dataset (using the cache we set up) and run the loop
        # We use debug=True which limits data to 100 samples
        run(debug=True, epochs=1, batch_size=2)

        print("[Check] Engine run completed without errors.")

        # Check if model was saved
        if os.path.exists(Config.BEST_MODEL_PATH):
            print(f"[Check] Best model saved at {Config.BEST_MODEL_PATH}")
        else:
            # It's possible validation didn't improve in 1 epoch, which is fine for a demo
            print("[Check] Run finished (Model might not have improved in 1 epoch).")

    except Exception as e:
        print(f"[Error] Engine run failed: {e}")
        raise e

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
