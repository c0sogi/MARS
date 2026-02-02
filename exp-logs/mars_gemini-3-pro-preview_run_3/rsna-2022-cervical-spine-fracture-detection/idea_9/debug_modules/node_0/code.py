import os
import sys
import pandas as pd
import torch
import numpy as np

# Import library components
from library.config import Config
from library.dataset import CervicalSpineDataset
from library.model import CervicalSpineMIL
from library.loss import HierarchicalCompoundLoss
from library.trainer import run_training
from library.inference import predict_test_set


def main():
    print("Initializing Demo...")

    # 1. Setup Reproducibility
    Config.setup_reproducibility(42)

    # 2. Create Demo Metadata Subsets
    # We create small subsets of the metadata to ensure the demo runs quickly.
    # We save these to the working directory.

    # Load original metadata
    # Note: We assume the metadata files exist as per the problem description
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create subsets (Top 4 for train, 2 for val, 2 for test)
    # This ensures we process very few DICOM directories
    demo_train = train_meta.head(4).copy()
    demo_val = val_meta.head(2).copy()
    demo_test = test_meta.head(2).copy()

    # Save to working dir
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    print(
        f"Created demo metadata: Train={len(demo_train)}, Val={len(demo_val)}, Test={len(demo_test)}"
    )

    # 3. Override Configuration for Speed
    print("Overriding Config for Demo...")
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    # Use a separate cache dir for demo to avoid conflicts or massive reads
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Reduce dimensions and compute load
    Config.IMAGE_SIZE = (128, 128)
    Config.SEQ_LENGTH = 16  # Reduced from 64
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # 4. Verify Dataset Logic
    print("\n--- Verifying Dataset ---")
    # load_cached_data=False forces the dataset to process DICOMs from scratch for these few items
    dataset = CervicalSpineDataset(
        demo_train_path, phase="train", load_cached_data=False
    )

    # Fetch one sample
    sample = dataset[0]

    # Check keys
    assert "image" in sample, "Sample missing 'image' key"
    assert "labels" in sample, "Sample missing 'labels' key"
    assert "study_id" in sample, "Sample missing 'study_id' key"

    # Check Image Shape: (Seq, Channels, H, W) -> (16, 3, 128, 128)
    img = sample["image"]
    expected_shape = (
        Config.SEQ_LENGTH,
        Config.IN_CHANNELS,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    assert (
        img.shape == expected_shape
    ), f"Unexpected image shape: {img.shape}, expected {expected_shape}"

    # Check Labels
    labels = sample["labels"]
    assert "vertebrae" in labels
    assert "patient_overall" in labels
    assert labels["vertebrae"].shape == (
        7,
    ), f"Unexpected vertebrae label shape: {labels['vertebrae'].shape}"

    print("Dataset verification passed.")

    # 5. Verify Model Logic
    print("\n--- Verifying Model ---")
    # Initialize model (pretrained=False to avoid downloading weights during demo)
    model = CervicalSpineMIL(pretrained=False)
    model.to(Config.DEVICE)

    # Create a dummy batch matching the dataset output
    # Unsqueeze to add batch dimension: (1, 16, 3, 128, 128)
    batch_imgs = img.unsqueeze(0).to(Config.DEVICE)

    # Forward pass
    outputs = model(batch_imgs)

    # Check outputs
    assert "vertebrae_logits" in outputs
    assert "patient_logit" in outputs
    assert outputs["vertebrae_logits"].shape == (
        1,
        7,
    ), "Vertebrae logits shape mismatch"
    assert outputs["patient_logit"].shape == (1, 1), "Patient logit shape mismatch"

    print("Model verification passed.")

    # 6. Verify Loss Logic
    print("\n--- Verifying Loss ---")
    criterion = HierarchicalCompoundLoss()

    # Create dummy targets
    targets = {
        "vertebrae": labels["vertebrae"].unsqueeze(0).to(Config.DEVICE),
        "patient_overall": labels["patient_overall"].unsqueeze(0).to(Config.DEVICE),
    }

    loss = criterion(outputs, targets)
    assert torch.is_tensor(loss), "Loss is not a tensor"
    assert loss.item() >= 0, "Loss is negative"
    print(f"Loss computed: {loss.item():.4f}")

    # 7. Run Training Pipeline
    print("\n--- Running Training Pipeline (Demo) ---")
    # We use load_cached_data=False to force the processing logic to run on our small subset
    # This tests the full end-to-end training loop including data loading, forward pass, backward pass, and saving.
    run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-3,
        patience=1,
        load_cached_data=False,
    )

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print(f"Training pipeline finished. Model saved to {Config.MODEL_SAVE_PATH}")

    # 8. Run Inference Pipeline
    print("\n--- Running Inference Pipeline (Demo) ---")
    # This tests loading the saved model, running inference on the test subset, and generating submission.csv
    predict_test_set(load_cached_data=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "row_id" in sub_df.columns, "Submission missing 'row_id' column"
    assert "fractured" in sub_df.columns, "Submission missing 'fractured' column"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"Inference pipeline finished. Submission rows: {len(sub_df)}")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
