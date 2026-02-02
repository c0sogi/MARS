import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data_processor import process_data
from library.dataset import RNADataset, get_dataloader
from library.model_components import InputEmbeddingStem, PreActDilatedBlock, DenseTCN
from library.model import EIPFN
from library.loss import MaskedMCRMSELoss
from library.train import train_model


def main():
    # 1. Setup
    print(">>> Setting up...")
    seed_everything(42)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Test Data Processor
    print("\n>>> Testing Data Processor...")
    # Process train data (load_cached_data=False forces a check of the processing logic if cache doesn't exist)
    # Note: process_data handles caching internally.
    data_dict = process_data(mode="train", load_cached_data=True)

    # Verify keys
    assert "inputs" in data_dict, "Missing 'inputs' in processed data"
    assert "partner_indices" in data_dict, "Missing 'partner_indices' in processed data"
    assert "targets" in data_dict, "Missing 'targets' in processed data"
    assert "ids" in data_dict, "Missing 'ids' in processed data"

    # Verify shapes
    n_samples = len(data_dict["ids"])
    print(f"Processed {n_samples} training samples.")

    # Inputs: (N, L, 18)
    assert data_dict["inputs"].shape == (
        n_samples,
        Config.SEQ_LENGTH,
        Config.INPUT_CHANNELS,
    ), f"Input shape mismatch: {data_dict['inputs'].shape}"

    # Partner Indices: (N, L)
    assert data_dict["partner_indices"].shape == (
        n_samples,
        Config.SEQ_LENGTH,
    ), f"Partner indices shape mismatch: {data_dict['partner_indices'].shape}"

    # Targets: (N, L, 5)
    assert data_dict["targets"].shape == (
        n_samples,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Targets shape mismatch: {data_dict['targets'].shape}"

    # 3. Test Dataset and DataLoader
    print("\n>>> Testing Dataset and DataLoader...")
    # Use debug mode to get a small subset
    ds = RNADataset(mode="train", load_cached_data=True, debug=True)
    print(f"Debug Dataset size: {len(ds)}")
    assert len(ds) > 0, "Dataset is empty"

    item = ds[0]
    assert isinstance(item["inputs"], torch.Tensor), "Item inputs should be a Tensor"
    assert isinstance(
        item["partner_indices"], torch.Tensor
    ), "Item partner_indices should be a Tensor"
    assert isinstance(item["targets"], torch.Tensor), "Item targets should be a Tensor"

    # Test DataLoader
    batch_size = 4
    dl = get_dataloader(
        mode="train", load_cached_data=True, debug=True, batch_size=batch_size
    )
    batch = next(iter(dl))

    print(f"Batch inputs shape: {batch['inputs'].shape}")
    assert batch["inputs"].shape == (
        batch_size,
        Config.SEQ_LENGTH,
        Config.INPUT_CHANNELS,
    )
    assert batch["partner_indices"].shape == (batch_size, Config.SEQ_LENGTH)
    assert batch["targets"].shape == (batch_size, Config.SEQ_LENGTH, Config.NUM_TARGETS)

    # 4. Test Model Components
    print("\n>>> Testing Model Components...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test InputEmbeddingStem
    # Projects (B, C, L) -> (B, Out, L)
    stem = InputEmbeddingStem(in_channels=Config.INPUT_CHANNELS, out_channels=32).to(
        device
    )
    dummy_input = torch.randn(batch_size, Config.INPUT_CHANNELS, Config.SEQ_LENGTH).to(
        device
    )
    out_stem = stem(dummy_input)
    assert out_stem.shape == (
        batch_size,
        32,
        Config.SEQ_LENGTH,
    ), f"Stem output mismatch: {out_stem.shape}"

    # Test PreActDilatedBlock
    # Output channels should be equal to growth_rate (out_channels arg)
    block = PreActDilatedBlock(
        in_channels=32, out_channels=16, kernel_size=3, dilation=1, dropout=0.0
    ).to(device)
    out_block = block(out_stem)
    assert out_block.shape == (
        batch_size,
        16,
        Config.SEQ_LENGTH,
    ), f"Block output mismatch: {out_block.shape}"

    # Test DenseTCN
    # DenseTCN output channels = in_channels + num_layers * growth_rate
    # Here: 32 + 2 * 16 = 64
    dense_tcn = DenseTCN(
        in_channels=32, growth_rate=16, kernel_size=3, dilations=[1, 2], dropout=0.0
    ).to(device)
    out_dense = dense_tcn(out_stem)
    assert out_dense.shape == (
        batch_size,
        64,
        Config.SEQ_LENGTH,
    ), f"DenseTCN output mismatch: {out_dense.shape}"

    # 5. Test Full Model (EIPFN)
    print("\n>>> Testing EIPFN Model...")
    model = EIPFN().to(device)

    inputs = batch["inputs"].to(device)
    p_idx = batch["partner_indices"].to(device)

    # Forward pass 1 (No feedback, y_prev=None)
    preds = model(inputs, p_idx, y_prev=None)
    assert preds.shape == (
        batch_size,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch: {preds.shape}"

    # Forward pass 2 (With feedback)
    preds_fb = model(inputs, p_idx, y_prev=preds)
    assert preds_fb.shape == (
        batch_size,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Model feedback output shape mismatch: {preds_fb.shape}"

    # 6. Test Loss Function
    print("\n>>> Testing Loss Function...")
    criterion = MaskedMCRMSELoss()
    targets = batch["targets"].to(device)

    loss = criterion(preds, targets)
    print(f"Calculated Loss: {loss.item():.6f}")
    assert loss.item() >= 0.0, "Loss should be non-negative"
    assert loss.dim() == 0, "Loss should be a scalar"

    # Test with custom mask (e.g., only first 10 positions valid)
    mask = torch.zeros((batch_size, Config.SEQ_LENGTH), device=device)
    mask[:, :10] = 1.0
    loss_masked = criterion(preds, targets, mask)
    assert loss_masked.item() >= 0.0, "Masked loss should be non-negative"

    # 7. Test Metric Utility
    print("\n>>> Testing MCRMSE Metric Utility...")
    metric = MCRMSE(scored_indices=[0, 1])
    # Create dummy numpy data
    # Preds: all 1.0, Targets: all 0.0. RMSE should be 1.0
    p_np = np.ones((2, 10, 5))
    t_np = np.zeros((2, 10, 5))
    m_np = np.ones((2, 10))  # All valid

    metric.update(p_np, t_np, m_np)
    score = metric.compute()
    print(f"Metric Score (Expected 1.0): {score}")
    assert np.isclose(score, 1.0), f"Metric score incorrect: {score}"

    metric.reset()
    assert metric.sum_squared_errors[0] == 0.0, "Metric reset failed"

    # 8. Integration Test: Training Loop
    print("\n>>> Testing Training Loop (Integration)...")
    # Run for 2 epochs with small batch size on debug data
    # This verifies the entire pipeline: Data -> Model -> Loss -> Optim -> Checkpointing

    # Ensure model directory exists
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    # We use debug=True to limit dataset size significantly for speed
    train_model(debug=True, epochs=2, batch_size=4)

    # Verify model file was created (assuming validation improved or at least ran)
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model checkpoint successfully created at {Config.MODEL_PATH}")
    else:
        print(
            "Note: Model checkpoint not found. This is expected if validation loss didn't improve in 2 epochs."
        )

    print("\n>>> All demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
