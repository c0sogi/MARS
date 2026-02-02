import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.dataset import InkDataset
from library.model import SegFormerMiTB2
from library.loss import BCEDiceLoss
from library.train import Trainer
from library.inference import InferenceEngine


def main():
    print("--- Starting Vesuvius Ink Detection Library Demo ---")

    # 1. Setup & Configuration Overrides for Speed
    print("\n[1] Configuring environment for fast demonstration...")

    # Set paths to working directory for temporary files
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters
    Config.CACHE_DIR = demo_dir
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2  # Small batch for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: Epochs=1, Batch=2, Cache=./working/demo_execution")

    # 2. Prepare Subset Metadata
    print("\n[2] Preparing subset metadata...")

    # Read original metadata
    full_train_df = pd.read_csv(Config.METADATA_TRAIN)
    full_val_df = pd.read_csv(Config.METADATA_VAL)

    # Create tiny subsets (4 train, 2 val)
    demo_train_df = full_train_df.head(4).copy()
    demo_val_df = full_val_df.head(2).copy()

    # Save to demo directory
    demo_train_path = os.path.join(demo_dir, "train.csv")
    demo_val_path = os.path.join(demo_dir, "validation.csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_val_df.to_csv(demo_val_path, index=False)

    # Point Config to these new files
    Config.METADATA_TRAIN = demo_train_path
    Config.METADATA_VAL = demo_val_path

    print(
        f"Subset metadata created: {len(demo_train_df)} train, {len(demo_val_df)} val samples."
    )

    # 3. Verify Dataset Logic
    print("\n[3] Verifying InkDataset...")
    dataset = InkDataset(demo_train_df, split="train", load_cached_data=True)

    # Check length
    assert len(dataset) == 4, f"Dataset length mismatch. Expected 4, got {len(dataset)}"

    # Check item structure
    img, label, mask = dataset[0]
    print(
        f"Dataset item shapes -> Image: {img.shape}, Label: {label.shape}, Mask: {mask.shape}"
    )

    # Assertions
    assert img.shape == (3, 512, 512), f"Unexpected image shape: {img.shape}"
    assert label.shape == (1, 512, 512), f"Unexpected label shape: {label.shape}"
    assert mask.shape == (1, 512, 512), f"Unexpected mask shape: {mask.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a torch Tensor"

    # 4. Verify Model Logic
    print("\n[4] Verifying SegFormerMiTB2 Model...")
    model = SegFormerMiTB2()
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        2,
        1,
        512,
        512,
    ), f"Unexpected model output shape: {output.shape}"

    # 5. Verify Loss Logic
    print("\n[5] Verifying BCEDiceLoss...")
    criterion = BCEDiceLoss()

    # Dummy logits (model output) and targets (binary mask)
    dummy_logits = torch.randn(2, 1, 512, 512)
    dummy_targets = torch.randint(0, 2, (2, 1, 512, 512)).float()

    loss = criterion(dummy_logits, dummy_targets)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # 6. Execute Training Loop
    print("\n[6] Executing Training Loop (1 Epoch)...")
    trainer = Trainer(debug=False, epochs=Config.EPOCHS)
    trainer.load_data()

    # Run training
    best_score = trainer.run()

    print(f"Training finished. Best Score: {best_score}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint was not saved."

    # 7. Execute Inference
    print("\n[7] Executing Inference...")

    # Ensure test metadata exists (using the provided one)
    assert os.path.exists(Config.METADATA_TEST), "Test metadata missing"

    # Run inference
    engine = InferenceEngine()
    engine.run(limit=1)  # Limit to 1 fragment to save time

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission file content:")
    print(sub_df.head())

    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission columns missing."
    assert len(sub_df) > 0, "Submission file is empty."

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
