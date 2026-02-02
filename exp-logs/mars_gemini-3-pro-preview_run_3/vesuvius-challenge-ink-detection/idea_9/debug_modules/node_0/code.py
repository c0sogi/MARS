import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import load_fragment_data, InkDataset, get_normalization_stats
from library.model import ESDN_PCH
from library.loss import BCEDiceLoss
from library.train_utils import train_one_epoch, evaluate, optimize_threshold
from library.inference_utils import rle_encode, predict_tiled


def run_demonstration():
    print("--- Starting Library Demonstration ---")

    # 1. Override Config for Speed
    # ----------------------------
    print("\n[1] Configuring environment for rapid testing...")
    Config.MAX_PATCHES_PER_EPOCH = 10  # Only use 10 patches per epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.EPOCHS = 1  # Only 1 epoch
    Config.PATCH_SIZE = 256  # Keep default patch size
    Config.Z_DIM = 65  # Keep default Z depth

    # Set seeds for reproducibility
    Config.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # 2. Data Loading & Dataset
    # -------------------------
    print("\n[2] Testing Data Loading...")

    # Test raw fragment loading
    # We use fragment '2' as it is available in the metadata provided in the prompt
    fragment_id = "2"
    print(f"    Loading fragment {fragment_id} metadata and cache...")
    try:
        data = load_fragment_data(fragment_id, split="train", load_cached_data=True)
        volume = data["volume"]
        mask = data["mask"]
        label = data["label"]

        print(f"    Volume shape: {volume.shape}")
        print(f"    Mask shape: {mask.shape}")

        assert (
            volume.shape[0] == Config.Z_DIM
        ), f"Expected Z_DIM={Config.Z_DIM}, got {volume.shape[0]}"
        assert (
            volume.shape[1:] == mask.shape
        ), "Volume (H, W) does not match Mask (H, W)"
        if label is not None:
            assert label.shape == mask.shape, "Label shape does not match Mask shape"
        print("    Fragment data loaded and verified.")
    except Exception as e:
        print(f"    Failed to load fragment data: {e}")
        # If real data is missing in this specific env, we proceed with synthetic checks where possible
        # But based on prompt, input exists.
        raise e

    # Test Dataset
    print("    Initializing InkDataset...")
    # We provide normalization stats manually to skip the expensive calculation over the whole dataset
    dummy_stats = (100.0, 20.0)

    dataset = InkDataset(
        split="train",
        fragment_ids=[fragment_id],
        mode="train",
        normalization_stats=dummy_stats,
    )

    print(f"    Dataset length (patches per epoch): {len(dataset)}")
    assert (
        len(dataset) == Config.MAX_PATCHES_PER_EPOCH
    ), "Dataset length mismatch with config override"

    # Test __getitem__
    sample_vol, sample_label = dataset[0]
    print(f"    Sample volume tensor shape: {sample_vol.shape}")
    print(f"    Sample label tensor shape: {sample_label.shape}")

    assert sample_vol.shape == (
        Config.Z_DIM,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect volume patch shape"
    assert sample_label.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect label patch shape"

    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Model Architecture
    # ---------------------
    print("\n[3] Testing Model Architecture...")
    model = ESDN_PCH().to(device)

    # Create dummy input: (Batch, Z, H, W)
    dummy_input = torch.randn(2, Config.Z_DIM, Config.PATCH_SIZE, Config.PATCH_SIZE).to(
        device
    )

    print("    Running forward pass on dummy input...")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model output shape: {output.shape}")
    assert output.shape == (
        2,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Model output shape mismatch"

    # 4. Loss Function
    # ----------------
    print("\n[4] Testing Loss Function (BCEDiceLoss)...")
    criterion = BCEDiceLoss(bce_weight=0.5)

    # Dummy logits (model output) and targets
    dummy_logits = torch.randn(2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE).to(device)
    dummy_targets = (
        torch.randint(0, 2, (2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE))
        .float()
        .to(device)
    )

    loss_val = criterion(dummy_logits, dummy_targets)
    print(f"    Calculated Loss: {loss_val.item():.4f}")
    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val > 0, "Loss should be positive"

    # 5. Training Loop Utils
    # ----------------------
    print("\n[5] Testing Training and Evaluation Loops...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train one epoch (using the small dataloader created earlier)
    print("    Running train_one_epoch...")
    avg_train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
    print(f"    Average Train Loss: {avg_train_loss:.4f}")

    # Evaluate
    print("    Running evaluate...")
    avg_val_loss, val_probs, val_labels = evaluate(model, dataloader, criterion, device)
    print(f"    Average Val Loss: {avg_val_loss:.4f}")
    print(f"    Validation Probs Size: {val_probs.size}")

    assert avg_train_loss > 0, "Train loss should be positive"
    assert len(val_probs) == len(val_labels), "Mismatch in validation outputs"

    # 6. Inference Utils
    # ------------------
    print("\n[6] Testing Inference Utilities...")

    # A. RLE Encoding
    # Create a simple 3x3 mask:
    # 0 1 0
    # 0 1 0
    # 0 0 0
    # Flattened (row-major): 0, 1, 0, 0, 1, 0, 0, 0, 0
    # Indices (1-based):     1, 2, 3, 4, 5, 6, 7, 8, 9
    # Ink at: 2 and 5.
    # Runs: Start 2, Len 1. Start 5, Len 1. -> "2 1 5 1"

    dummy_mask = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)

    rle_str = rle_encode(dummy_mask)
    print(f"    RLE Output: '{rle_str}'")
    assert (
        rle_str == "2 1 5 1"
    ), f"RLE Encoding failed. Expected '2 1 5 1', got '{rle_str}'"

    # B. Threshold Optimization
    # Create synthetic probs and labels
    # Probs: [0.1, 0.4, 0.6, 0.9]
    # Labels:[ 0,   0,   1,   1 ]
    # Best threshold should be around 0.5
    syn_probs = np.array([0.1, 0.4, 0.6, 0.9])
    syn_labels = np.array([0, 0, 1, 1])

    best_thresh, best_score = optimize_threshold(syn_probs, syn_labels)
    print(f"    Optimized Threshold: {best_thresh:.2f}, Score: {best_score:.4f}")
    assert (
        0.4 < best_thresh <= 0.6
    ), f"Threshold optimization gave unexpected result: {best_thresh}"

    # C. Tiled Prediction
    # We create a synthetic volume larger than patch size (256) but small enough to be fast.
    # Size: (65, 300, 300)
    print("    Testing predict_tiled with synthetic volume...")
    H, W = 300, 300
    syn_vol = np.random.randint(0, 255, (Config.Z_DIM, H, W), dtype=np.uint8)
    syn_mask = np.ones((H, W), dtype=np.uint8)  # Full valid mask

    # Run tiled prediction
    # We use the model we initialized earlier
    # Pass dummy stats
    pred_map = predict_tiled(
        model, syn_vol, syn_mask, mean=100.0, std=20.0, device=device, overlap_ratio=0.5
    )

    print(f"    Prediction Map Shape: {pred_map.shape}")
    assert pred_map.shape == (H, W), f"Expected shape ({H}, {W}), got {pred_map.shape}"
    assert pred_map.dtype == np.float32, "Prediction map should be float32"
    assert (
        pred_map.min() >= 0.0 and pred_map.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
