import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything, rle_encoding, fbeta_score
from library.data import InkDataset
from library.model import WideContextSegFormer
from library.loss import BCEDiceLoss
from library.engine import train_one_epoch
from library.inference import predict_fragment


def run_demo():
    print("--- Starting Vesuvius Ink Detection Demo ---")

    # 1. Setup & Configuration Override
    # Set seed for reproducibility
    seed_everything(CFG.seed)

    # Override CFG parameters for a fast demonstration
    CFG.epochs = 1
    CFG.batch_size = 2  # Small batch size for demo
    CFG.working_dir = "./working/demo_run"
    CFG.model_path = os.path.join(CFG.working_dir, "demo_model.pth")
    CFG.submission_path = os.path.join(CFG.working_dir, "submission.csv")

    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    print(f"Device: {CFG.device}")
    print(f"Working Directory: {CFG.working_dir}")

    # 2. Verify Utility Functions
    print("\n--- Verifying Utilities ---")

    # Test RLE Encoding
    # Mask: [[0, 1, 1, 0], [0, 0, 1, 0]]
    # Flattened: 0, 1, 1, 0, 0, 0, 1, 0
    # Indices (1-based): 2, 3 (run 1); 7 (run 2)
    # Expected RLE: "2 2 7 1" (Start 2, Len 2; Start 7, Len 1)
    dummy_mask = np.array([[0, 1, 1, 0], [0, 0, 1, 0]], dtype=np.uint8)
    rle_result = rle_encoding(dummy_mask)
    print(f"RLE Test Result: {rle_result}")
    assert (
        rle_result == "2 2 7 1"
    ), f"RLE verification failed. Expected '2 2 7 1', got '{rle_result}'"

    # Test F-beta Score
    # Preds: 0.1 (TN), 0.9 (TP), 0.8 (TP), 0.2 (TN) -> Binary: 0, 1, 1, 0
    # Targets: 0, 1, 1, 0
    dummy_preds = torch.tensor([0.1, 0.9, 0.8, 0.2])
    dummy_targets = torch.tensor([0, 1, 1, 0])
    score = fbeta_score(dummy_preds, dummy_targets, beta=0.5, threshold=0.5)
    print(f"F0.5 Score Test Result: {score}")
    assert (
        abs(score - 1.0) < 1e-5
    ), "F-beta score verification failed for perfect match."

    # 3. Data Loading Demonstration
    print("\n--- Verifying Data Loading ---")

    if not os.path.exists(CFG.train_metadata_path):
        raise FileNotFoundError(
            f"Training metadata not found at {CFG.train_metadata_path}"
        )

    # Load metadata and subset for speed
    full_train_df = pd.read_csv(CFG.train_metadata_path)
    print(f"Full training samples: {len(full_train_df)}")

    # Use only 4 samples for the demo
    demo_train_df = full_train_df.head(4).copy()

    # Instantiate Dataset
    # load_cached_data=True will process volumes and save .npy to CFG.working_dir
    print("Initializing InkDataset (this may take a moment to load/cache volumes)...")
    train_ds = InkDataset(demo_train_df, mode="train", load_cached_data=True)

    # Instantiate DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for simple demo script
        drop_last=False,
    )

    # Fetch one batch to verify shapes
    images, masks, indices = next(iter(train_loader))
    print(f"Batch Shapes -> Images: {images.shape}, Masks: {masks.shape}")

    # Assertions for shapes
    # Images: (Batch, Channels=5, Height=512, Width=512)
    assert images.shape == (
        CFG.batch_size,
        5,
        CFG.image_size,
        CFG.image_size,
    ), "Image batch shape incorrect."
    # Masks: (Batch, 1, Height=512, Width=512)
    assert masks.shape == (
        CFG.batch_size,
        1,
        CFG.image_size,
        CFG.image_size,
    ), "Mask batch shape incorrect."
    assert images.dtype == torch.float32, "Images should be float32."

    # 4. Model & Training Demonstration
    print("\n--- Verifying Model & Training ---")

    # Initialize Model
    model = WideContextSegFormer(CFG)
    model.to(CFG.device)

    # Check Forward Pass
    with torch.no_grad():
        outputs = model(images.to(CFG.device))
    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        CFG.batch_size,
        1,
        CFG.image_size,
        CFG.image_size,
    ), "Model output shape mismatch."

    # Initialize Loss and Optimizer
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate)

    # Train for one epoch
    print("Running training for 1 epoch on subset...")
    epoch_loss = train_one_epoch(model, optimizer, train_loader, CFG.device, criterion)
    print(f"Epoch Loss: {epoch_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), CFG.model_path)
    print(f"Model checkpoint saved to {CFG.model_path}")
    assert os.path.exists(CFG.model_path), "Model file was not created."

    # 5. Inference Demonstration
    print("\n--- Verifying Inference ---")

    if os.path.exists(CFG.test_metadata_path):
        test_df = pd.read_csv(CFG.test_metadata_path)
        fragment_ids = test_df["fragment_id"].unique()

        if len(fragment_ids) > 0:
            fid = fragment_ids[0]
            print(f"Processing test fragment: {fid}")

            # Predict on the fragment
            # We limit debug_max_batches to 2 to ensure the demo finishes quickly
            # In a real run, this parameter would be None to process the full image
            binary_mask = predict_fragment(
                fid, test_df, model, CFG.device, debug_max_batches=2
            )

            print(f"Inference Mask Shape: {binary_mask.shape}")
            assert binary_mask.ndim == 2, "Inference output should be a 2D mask."

            # Encode prediction
            rle = rle_encoding(binary_mask)

            # Create Submission
            submission_df = pd.DataFrame([{"Id": fid, "Predicted": rle}])
            submission_df.to_csv(CFG.submission_path, index=False)
            print(f"Submission saved to {CFG.submission_path}")

            # Validate Submission File
            saved_df = pd.read_csv(CFG.submission_path)
            assert len(saved_df) == 1, "Submission file should have 1 row."
            assert list(saved_df.columns) == [
                "Id",
                "Predicted",
            ], "Submission columns mismatch."

        else:
            print("No fragments found in test metadata.")
    else:
        print("Test metadata file not found. Skipping inference step.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
