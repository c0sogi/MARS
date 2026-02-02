import os
import shutil
import torch
import numpy as np
import pandas as pd
import time
from library.config import Config
from library.utils import set_seed, MCRMSE
from library.data import get_structure_adj, preprocess_data, get_dataloaders, RNADataset
from library.model import DeepDecoupledModel, StructuralInteractionModule
from library.train import train_one_epoch, validate


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("Initializing configuration...")

    # Create a demo configuration based on the provided Config class
    # We override specific parameters to ensure the demo runs quickly
    config = Config()
    config.working_dir = "./working/demo_execution"
    config.model_save_path = os.path.join(config.working_dir, "demo_model.pth")

    # Speed optimizations for demo
    config.num_epochs = 1
    config.batch_size = 16
    config.hidden_dim = 64  # Reduced from 384
    config.num_layers = 2  # Reduced from 4
    config.conv_filters = 32  # Reduced from 256
    config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    if os.path.exists(config.working_dir):
        shutil.rmtree(config.working_dir)
    os.makedirs(config.working_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.seed)
    device = torch.device(config.device)
    print(f"Configuration set. Device: {device}")

    # ==========================================
    # 2. Verify Utility Functions (Structure Parsing)
    # ==========================================
    print("\nVerifying structure adjacency parsing...")

    # Test case: "((..))" -> Length 6
    # Indices: 0-5. 0 pairs with 5, 1 pairs with 4. 2 and 3 are unpaired.
    dummy_struct = "((..))"
    seq_len = 6
    pair_indices, pair_mask = get_structure_adj(dummy_struct, seq_len)

    # Expected: [5, 4, 2, 3, 1, 0]
    expected_indices = np.array([5, 4, 2, 3, 1, 0])
    # Expected Mask: [1, 1, 0, 0, 1, 1]
    expected_mask = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    np.testing.assert_array_equal(
        pair_indices, expected_indices, err_msg="Structure indices mismatch"
    )
    np.testing.assert_array_equal(
        pair_mask, expected_mask, err_msg="Structure mask mismatch"
    )
    print("Structure parsing logic verified.")

    # ==========================================
    # 3. Verify Data Preprocessing
    # ==========================================
    print("\nVerifying data preprocessing...")

    # Create a dummy DataFrame mimicking train.parquet
    dummy_seq = "A" * 107
    dummy_struct = "." * 107
    dummy_loop = "E" * 107

    # Targets: 5 columns, length 68
    # We'll use random values for the target columns
    dummy_data = {
        "id": ["id_test_01", "id_test_02"],
        "sequence": [dummy_seq, dummy_seq],
        "structure": [dummy_struct, dummy_struct],
        "predicted_loop_type": [dummy_loop, dummy_loop],
        "reactivity": [[0.1] * 68, [0.2] * 68],
        "deg_Mg_pH10": [[0.1] * 68, [0.2] * 68],
        "deg_pH10": [[0.1] * 68, [0.2] * 68],
        "deg_Mg_50C": [[0.1] * 68, [0.2] * 68],
        "deg_50C": [[0.1] * 68, [0.2] * 68],
    }
    df_dummy = pd.DataFrame(dummy_data)

    inputs, p_indices, p_mask, targets, ids = preprocess_data(
        df_dummy, config, is_test=False
    )

    # Assert Shapes
    # Inputs: (N, 107, 14)
    assert inputs.shape == (2, 107, 14), f"Input shape mismatch: {inputs.shape}"
    # Pair Indices: (N, 107)
    assert p_indices.shape == (
        2,
        107,
    ), f"Pair indices shape mismatch: {p_indices.shape}"
    # Targets: (N, 68, 5)
    assert targets.shape == (2, 68, 5), f"Target shape mismatch: {targets.shape}"

    print("Data preprocessing shapes verified.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\nVerifying model architecture...")

    # Instantiate model with demo config
    # We need to temporarily patch Config inside the model module or pass config,
    # but the provided DeepDecoupledModel instantiates Config() internally.
    # Since Config is a class with class attributes in the provided library,
    # modifying the class attributes globally (as we did in step 1) works
    # because the model re-instantiates Config() which reads the modified class attrs.

    model = DeepDecoupledModel().to(device)

    # Create dummy tensors on device
    # Batch size 4, Seq Len 107, Features 14
    dummy_x = torch.randn(4, 107, 14).to(device)
    dummy_pidx = torch.zeros(4, 107, dtype=torch.long).to(device)
    dummy_pmask = torch.zeros(4, 107).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_x, dummy_pidx, dummy_pmask)

    # Expected output: (Batch, SeqLen, Targets) -> (4, 107, 5)
    # Note: The model outputs predictions for the full sequence length (107),
    # slicing happens in the loss function.
    assert output.shape == (4, 107, 5), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass verified.")

    # ==========================================
    # 5. Verify Metric (MCRMSE)
    # ==========================================
    print("\nVerifying MCRMSE metric...")

    criterion = MCRMSE()

    # Case 1: Perfect prediction
    # Preds: (1, 68, 5), Trues: (1, 68, 5)
    t_preds = torch.ones(1, 68, 5).to(device)
    t_trues = torch.ones(1, 68, 5).to(device)

    loss_val = criterion(t_preds, t_trues, mode="train")
    assert torch.isclose(
        loss_val, torch.tensor(0.0).to(device)
    ), "Metric should be 0 for perfect match"

    # Case 2: Known error
    # Preds = 1.0, Trues = 0.0 -> Diff = 1.0 -> Sq = 1.0 -> RMSE = 1.0 -> Mean = 1.0
    t_preds = torch.ones(1, 68, 5).to(device)
    t_trues = torch.zeros(1, 68, 5).to(device)

    loss_val = criterion(t_preds, t_trues, mode="train")
    assert torch.isclose(
        loss_val, torch.tensor(1.0).to(device)
    ), f"Metric should be 1.0, got {loss_val}"

    # Case 3: Validation mode (Scored columns only)
    # Scored indices are [0, 1, 3].
    # Let's make error 0 on scored columns and 1 on others.
    # Col 0, 1, 3 -> Pred 0, True 0 (Error 0)
    # Col 2, 4    -> Pred 1, True 0 (Error 1)
    t_preds = torch.zeros(1, 68, 5).to(device)
    t_preds[:, :, [2, 4]] = 1.0
    t_trues = torch.zeros(1, 68, 5).to(device)

    loss_val = criterion(t_preds, t_trues, mode="val")
    # Should be 0 because we ignore cols 2 and 4 in val mode
    assert torch.isclose(
        loss_val, torch.tensor(0.0).to(device)
    ), f"Val metric should ignore unscored columns, got {loss_val}"

    print("Metric calculation verified.")

    # ==========================================
    # 6. Full Training Loop Execution
    # ==========================================
    print("\nExecuting training loop demo...")

    # Load real data (using the metadata files provided in the environment)
    # This uses the cached logic in library.data.load_or_process_data
    # We force reload to ensure we use the 'demo_execution' working dir cache
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    print(f"Training for {config.num_epochs} epoch(s) on device {device}...")

    # Train
    start_time = time.time()
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, config
    )
    end_time = time.time()

    print(f"Epoch 1 Train Loss: {train_loss:.4f} (Time: {end_time - start_time:.2f}s)")

    # Validate
    val_score = validate(model, val_loader, criterion, device, config)
    print(f"Validation MCRMSE: {val_score:.4f}")

    # Save Model
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved to {config.model_save_path}")

    # Check if file exists
    assert os.path.exists(config.model_save_path), "Model file was not saved."

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
