import sys
import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Ensure the current directory is in the path to import the library modules
sys.path.append(os.getcwd())

# Import provided library modules
import library.config as lib_config
import library.utils as lib_utils
import library.data as lib_data
import library.model as lib_model
import library.engine as lib_engine


def main():
    print("=== Starting Library Usage Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Define a temporary working directory for this demo to isolate outputs
    demo_work_dir = "./working/demo_execution/"
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)
    os.makedirs(demo_work_dir, exist_ok=True)

    # Monkeypatch the global Config to use our demo directory
    # This ensures process_data saves/loads from here
    lib_config.Config.WORK_DIR = demo_work_dir

    # Define a lightweight configuration for rapid execution
    class DemoConfig:
        # Paths
        WORK_DIR = demo_work_dir

        # Model Architecture (Reduced for speed)
        HIDDEN_DIM = 32
        NUM_LAYERS = 2
        NHEAD = 4  # Must be a divisor of HIDDEN_DIM
        DROPOUT = 0.0

        # Training Hyperparameters
        BATCH_SIZE = 16
        EPOCHS = 1
        LR = 1e-3

        # Loss Parameters
        LABEL_SMOOTHING = 0.0
        RECON_LAMBDA = 0.5
        MASK_PROB = 0.15

        # Hardware
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds for reproducibility
    lib_utils.seed_everything(42)
    print(f"    Working Directory: {DemoConfig.WORK_DIR}")
    print(f"    Device: {DemoConfig.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Data Processing
    # --------------------------------------------------------------------------
    print("\n[2] Processing Data (library.data.process_data)...")

    # Run the data processing pipeline.
    # This reads metadata/train.csv, engineers features, tokenizes, and scales.
    # It returns a dictionary of numpy arrays and the vocab size.
    data, vocab_size = lib_data.process_data(load_cached_data=True)

    # Verify output keys
    required_keys = [
        "X_num_train",
        "X_seq_train",
        "y_train",
        "X_num_val",
        "X_seq_val",
        "y_val",
        "X_num_test",
        "X_seq_test",
        "ids_test",
    ]
    for key in required_keys:
        if key not in data:
            raise AssertionError(f"Missing key in processed data: {key}")
        if not isinstance(data[key], np.ndarray):
            raise AssertionError(f"Data for {key} is not a numpy array")

    print(f"    Data processing complete.")
    print(f"    Vocab Size: {vocab_size}")
    print(f"    Training Samples: {len(data['X_num_train'])}")

    # --------------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Dataset (library.data.ManufacturingDataset)...")

    # Create a small subset (100 samples) to ensure the rest of the demo runs instantly
    subset_size = 100
    X_num_sub = data["X_num_train"][:subset_size]
    X_seq_sub = data["X_seq_train"][:subset_size]
    y_sub = data["y_train"][:subset_size]

    # Instantiate Dataset
    dataset = lib_data.ManufacturingDataset(
        X_num=X_num_sub, X_seq=X_seq_sub, y=y_sub, mask_prob=DemoConfig.MASK_PROB
    )

    # Verify __getitem__
    sample = dataset[0]
    expected_item_keys = ["x_num", "x_seq", "target_seq", "mask_seq", "target"]
    for k in expected_item_keys:
        if k not in sample:
            raise AssertionError(f"Dataset sample missing key: {k}")
        if not torch.is_tensor(sample[k]):
            raise AssertionError(f"Dataset item '{k}' is not a tensor")

    print("    Dataset item structure verified.")

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=DemoConfig.BATCH_SIZE, shuffle=True)
    batch = next(iter(loader))
    print(f"    DataLoader batch generated. x_num shape: {batch['x_num'].shape}")

    # --------------------------------------------------------------------------
    # 4. Model Instantiation & Forward Pass
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model (library.model.ResDeGUT)...")

    num_features = data["X_num_train"].shape[1]
    seq_len = data["X_seq_train"].shape[1]

    # Instantiate Model with DemoConfig
    model = lib_model.ResDeGUT(
        num_features=num_features,
        seq_len=seq_len,
        vocab_size=vocab_size,
        config=DemoConfig,
    ).to(DemoConfig.DEVICE)

    print("    Model instantiated.")

    # Run Forward Pass
    x_num_b = batch["x_num"].to(DemoConfig.DEVICE)
    x_seq_b = batch["x_seq"].to(DemoConfig.DEVICE)

    with torch.no_grad():
        logits, recon_logits = model(x_num_b, x_seq_b)

    # Verify Output Shapes
    # Logits should be [Batch, 1]
    if logits.shape != (x_num_b.size(0), 1):
        raise AssertionError(
            f"Logits shape mismatch. Expected {(x_num_b.size(0), 1)}, got {logits.shape}"
        )

    # Recon Logits should be [Batch, SeqLen, VocabSize]
    # The model reconstructs the last `seq_len` tokens
    if recon_logits.shape != (x_num_b.size(0), seq_len, vocab_size):
        raise AssertionError(
            f"Recon logits shape mismatch. Expected {(x_num_b.size(0), seq_len, vocab_size)}, got {recon_logits.shape}"
        )

    print("    Forward pass successful. Output shapes correct.")

    # --------------------------------------------------------------------------
    # 5. Engine Execution (Train, Eval, Predict)
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Engine (library.engine)...")

    optimizer = AdamW(model.parameters(), lr=DemoConfig.LR)
    scheduler = None  # Not strictly needed for this short demo

    # Test Training Function
    print("    Running train_fn...")
    train_loss = lib_engine.train_fn(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DemoConfig.DEVICE,
        config=DemoConfig,
    )
    print(f"    Train Loss: {train_loss:.4f}")

    if not np.isfinite(train_loss):
        raise AssertionError("Training loss is not finite/valid.")

    # Test Evaluation Function
    print("    Running eval_fn...")
    val_loss, val_auc = lib_engine.eval_fn(
        model=model,
        data_loader=loader,  # Using training loader just for mechanics check
        device=DemoConfig.DEVICE,
    )
    print(f"    Eval Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Test Prediction Function
    print("    Running predict_fn...")
    # Create a test loader (no targets)
    test_dataset = lib_data.ManufacturingDataset(
        X_num=X_num_sub, X_seq=X_seq_sub, y=None, mask_prob=0.0
    )
    test_loader = DataLoader(test_dataset, batch_size=DemoConfig.BATCH_SIZE)

    preds = lib_engine.predict_fn(
        model=model, data_loader=test_loader, device=DemoConfig.DEVICE
    )

    if len(preds) != subset_size:
        raise AssertionError(
            f"Prediction count mismatch. Expected {subset_size}, got {len(preds)}"
        )

    if preds.min() < 0.0 or preds.max() > 1.0:
        raise AssertionError("Predictions are not valid probabilities (0-1).")

    print("    Predictions generated successfully.")

    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
