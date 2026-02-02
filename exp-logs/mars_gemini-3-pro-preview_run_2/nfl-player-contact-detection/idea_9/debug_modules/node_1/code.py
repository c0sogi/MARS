import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_processing import DataProcessor
from library.dataset import NFLDataset
from library.model import WIRKNet
from library.training import FocalLoss, train_epoch, evaluate, optimize_threshold


def run_demo():
    print("--- Setting up Configuration for Demo ---")

    # 1. Configure for Speed and Isolation
    # Create a dedicated working directory for this demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to point to the demo directory
    # This prevents overwriting any real training artifacts
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_FEATURES_PATH = os.path.join(demo_dir, "train_features_debug.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(demo_dir, "val_features_debug.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(demo_dir, "test_features_debug.parquet")
    Config.SCALER_PATH = os.path.join(demo_dir, "scaler_debug.joblib")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model_debug.pth")

    # Enable Debug mode to reduce data size for Train/Val processing
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Process only 2000 rows for speed
    Config.BATCH_SIZE = 32
    Config.MAX_EPOCHS = 1

    # Create a subset of Test Metadata to speed up the inference step
    # The original test set is large, so we create a 500-row subset
    print("Creating test metadata subset...")
    original_test_meta = pd.read_csv(Config.TEST_META_PATH)
    test_subset_path = os.path.join(demo_dir, "test_meta_subset.csv")
    original_test_meta.head(500).to_csv(test_subset_path, index=False)
    Config.TEST_META_PATH = test_subset_path  # Override path to use subset

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print("\n--- 1. Data Processing (Train/Val) ---")
    processor = DataProcessor()

    # Force processing from scratch (load_cached_data=False) to demonstrate the pipeline logic
    # This will trigger the DEBUG sampling logic inside DataProcessor
    X_train, y_train, X_val, y_val = processor.get_train_val_data(
        load_cached_data=False
    )

    # Verification
    print(f"Train features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")

    assert not X_train.empty, "X_train should not be empty"
    assert not X_val.empty, "X_val should not be empty"
    assert len(X_train) == len(y_train), "Mismatch in training features and labels"
    assert processor.is_fitted, "Processor should be fitted after processing train data"

    # Verify feature engineering (Wide format check)
    # We expect columns for lags (e.g., x_position_1_lag-1) or flattened features
    assert any("x_position" in c for c in X_train.columns), "Kinematic features missing"
    assert "position_1_enc" in X_train.columns, "Encoded categorical feature missing"

    print("\n--- 2. Dataset Instantiation ---")
    train_dataset = NFLDataset(X_train, y_train)
    val_dataset = NFLDataset(X_val, y_val)

    # Verification of Dataset behavior
    assert len(train_dataset) == len(X_train)
    sample_cat, sample_cont, sample_y = train_dataset[0]

    print(
        f"Sample shapes -> Cat: {sample_cat.shape}, Cont: {sample_cont.shape}, Target: {sample_y.shape}"
    )

    assert (
        torch.is_tensor(sample_cat) and sample_cat.dtype == torch.long
    ), "Categorical features must be LongTensor"
    assert (
        torch.is_tensor(sample_cont) and sample_cont.dtype == torch.float32
    ), "Continuous features must be FloatTensor"
    assert sample_y.shape == (1,), "Target shape must be (1,)"

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print("\n--- 3. Model Initialization ---")
    # Determine input dimensions dynamically from the dataset
    num_cont_features = train_dataset.X_cont.shape[1]
    num_cat_features = train_dataset.X_cat.shape[1]

    # Calculate vocab sizes for embeddings
    # We inspect the max index in the training data to determine required embedding size
    if num_cat_features > 0:
        cat_vocab_sizes = []
        for i in range(num_cat_features):
            max_id = train_dataset.X_cat[:, i].max().item()
            cat_vocab_sizes.append(max_id + 1)
    else:
        cat_vocab_sizes = []

    print(
        f"Model Input Config: {num_cont_features} continuous, {num_cat_features} categorical features"
    )

    model = WIRKNet(
        num_cont_features=num_cont_features,
        cat_vocab_sizes=cat_vocab_sizes,
        hidden_dim=128,  # Reduced hidden dim for demo speed
        num_blocks=1,  # Reduced depth for demo speed
    )

    device = torch.device("cpu")  # Use CPU for this lightweight demo
    model.to(device)

    # Verify Forward Pass with a single batch
    with torch.no_grad():
        dummy_cat = sample_cat.unsqueeze(0).to(device)
        dummy_cont = sample_cont.unsqueeze(0).to(device)
        logits = model(dummy_cat, dummy_cont)
        assert logits.shape == (
            1,
            1,
        ), f"Output shape mismatch. Expected (1, 1), got {logits.shape}"
        print("Forward pass verification successful.")

    print("\n--- 4. Training Loop Execution ---")
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Run one training epoch
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss resulted in NaN"

    # Run evaluation on validation set
    val_loss, val_probs, val_targets = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    assert len(val_probs) == len(val_dataset), "Probability count mismatch"
    assert (
        val_probs.min() >= 0 and val_probs.max() <= 1
    ), "Probabilities out of range [0, 1]"

    # Optimize threshold based on validation results
    best_thresh, best_mcc = optimize_threshold(val_targets, val_probs)
    print(f"Best Threshold: {best_thresh:.2f}, Validation MCC: {best_mcc:.4f}")

    print("\n--- 5. Inference (Test Data) ---")
    # Process test data (using the subset created earlier)
    # The processor uses the scaler fitted on training data
    X_test, ids_test = processor.get_test_data(load_cached_data=False)

    print(f"Test Feature shape: {X_test.shape}")
    assert len(X_test) == 500, "Test subset size mismatch (expected 500)"
    assert not X_test.isnull().values.any(), "Test features contain NaNs"

    # Create Test Dataset (Note: y=None for inference)
    test_dataset = NFLDataset(X_test, y=None)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run Inference
    model.eval()
    test_preds = []
    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    assert len(test_preds) == len(X_test), "Prediction count mismatch"

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "contact_id": ids_test,
            "contact": (test_preds >= best_thresh).astype(int).flatten(),
        }
    )

    print("Sample Submission:")
    print(submission.head())

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
