import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coefficient, AverageMeter
from library.data_loader import get_data_loaders
from library.architecture import GC_ConvNeXtUNet
from library.losses import HybridLoss


def main():
    print("Starting demonstration of Contrail Identification pipeline...")

    # 1. Setup and Configuration
    # --------------------------
    print("\n[1] Initializing Configuration and Seeding...")
    Config.display()
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Verify Utility Functions
    # ---------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a simple 4x4 mask:
    # 0 0 0 0
    # 1 1 0 0  -> Pixels 2, 6 (1-based, column-major flattened: 1,2,3,4 | 5,6,7,8...)
    # 0 0 0 0     Wait, flattened column-major (Fortran):
    # 0 0 0 0     Col 1: 0,1,0,0 -> Pixel 2 is 1
    #             Col 2: 0,1,0,0 -> Pixel 6 is 1
    #             Col 3: 0,0,0,0
    #             Col 4: 0,0,0,0
    # Indices: 2, 6.
    # RLE logic: 'start length'.
    # Run 1: start 2, len 1. Run 2: start 6, len 1. -> "2 1 6 1"

    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[1, 0] = 1  # Row 1, Col 0 (Pixel 2)
    dummy_mask[1, 1] = 1  # Row 1, Col 1 (Pixel 6)

    encoded = rle_encode(dummy_mask)
    print(f"  Dummy Mask RLE: {encoded}")
    assert (
        encoded == "2 1 6 1"
    ), f"RLE Encoding failed. Expected '2 1 6 1', got '{encoded}'"

    # Test Dice Coefficient
    y_true = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    y_pred = torch.tensor(
        [1, 0, 1, 0], dtype=torch.float32
    )  # Intersection: 1 (first pixel), Union: 2+2=4
    # Dice = 2*1 / (2+2) = 0.5
    dice = dice_coefficient(y_pred, y_true)
    print(f"  Dummy Dice Score: {dice:.4f}")
    assert abs(dice - 0.5) < 1e-5, f"Dice calculation failed. Expected 0.5, got {dice}"

    # 3. Data Loading
    # ---------------
    print("\n[3] Loading Data (Debug Mode)...")
    # Use debug mode to load a tiny subset quickly
    train_loader, val_loader, test_loader = get_data_loaders(
        debug=True,
        debug_sample_size=10,
        batch_size=2,  # Small batch size for demo
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )

    print(f"  Train Loader batches: {len(train_loader)}")
    print(f"  Val Loader batches: {len(val_loader)}")
    print(f"  Test Loader batches: {len(test_loader)}")

    # Fetch one batch
    images, metadata, masks = next(iter(train_loader))

    print(f"  Batch Shapes:")
    print(f"    Images:   {images.shape}")  # Expected: (B, 6, 256, 256)
    print(f"    Metadata: {metadata.shape}")  # Expected: (B, 3)
    print(f"    Masks:    {masks.shape}")  # Expected: (B, 1, 256, 256)

    assert images.shape[1] == Config.IN_CHANNELS, "Incorrect input channels"
    assert metadata.shape[1] == Config.METADATA_FEATURE_DIM, "Incorrect metadata dim"
    assert masks.shape[1] == 1, "Incorrect mask channels"

    # 4. Model Initialization & Forward Pass
    # --------------------------------------
    print("\n[4] Initializing Model and Forward Pass...")
    device = Config.DEVICE
    print(f"  Device: {device}")

    model = GC_ConvNeXtUNet(
        in_chans=Config.IN_CHANNELS,
        num_classes=1,
        metadata_dim=Config.METADATA_FEATURE_DIM,
    ).to(device)

    # Move batch to device
    images = images.to(device)
    metadata = metadata.to(device)
    masks = masks.to(device)

    # Forward pass
    logits = model(images, metadata)
    print(f"  Output Logits Shape: {logits.shape}")

    assert logits.shape == masks.shape, "Model output shape mismatch with target"

    # 5. Loss Calculation & Optimization
    # ----------------------------------
    print("\n[5] Computing Loss and Gradients...")
    criterion = HybridLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Compute Loss
    loss = criterion(logits, masks)
    print(f"  Loss Value: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  Optimization step completed successfully.")

    # 6. Inference and Submission Generation
    # --------------------------------------
    print("\n[6] Simulating Inference on Test Data...")

    # Get a batch from test loader
    # Test loader returns: image, metadata, record_id
    try:
        test_images, test_meta, record_ids = next(iter(test_loader))
    except StopIteration:
        print(
            "  Test loader is empty (likely due to debug sampling). Creating dummy test batch."
        )
        test_images = torch.randn(2, 6, 256, 256)
        test_meta = torch.randn(2, 3)
        record_ids = ["100001", "100002"]

    test_images = test_images.to(device)
    test_meta = test_meta.to(device)

    model.eval()
    with torch.no_grad():
        test_logits = model(test_images, test_meta)
        test_probs = torch.sigmoid(test_logits)

        # Thresholding
        test_preds = (test_probs > 0.5).float().cpu().numpy()

    # Generate RLEs
    submission_data = []
    print("  Encoding predictions...")
    for i, rid in enumerate(record_ids):
        # Extract mask (H, W) - remove channel dim
        pred_mask = test_preds[i, 0, :, :]
        rle = rle_encode(pred_mask)
        submission_data.append({"record_id": str(rid), "encoded_pixels": rle})
        print(f"    ID: {rid}, RLE Length: {len(rle)}")

    # Create DataFrame
    sub_df = pd.DataFrame(submission_data)

    # Save to working directory
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    sub_df.to_csv(output_path, index=False)
    print(f"  Submission saved to: {output_path}")

    # Verify file existence
    assert os.path.exists(output_path), "Submission file was not created."

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
