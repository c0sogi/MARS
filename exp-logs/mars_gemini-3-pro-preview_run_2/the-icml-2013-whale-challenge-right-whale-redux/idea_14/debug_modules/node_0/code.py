import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_data_loaders
from library.models import get_model
from library.engine import fit_model, predict
from library.stacking import train_meta_learner, generate_meta_submission


def run_demo():
    print("=== Starting Right Whale Detection Demo ===")

    # 1. Setup and Configuration
    # We modify the Config class directly to optimize for a quick demo run.
    print("[1] Configuring environment for demo...")
    seed_everything(42)

    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_FOLDS = 2  # Use 2 folds so split is 50/50
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # 2. Create Mini Metadata Sets
    # We create small subsets of the original metadata to speed up data loading.
    print("[2] Creating mini-datasets...")

    # Load original metadata
    train_df_orig = pd.read_csv("./metadata/train.csv")
    val_df_orig = pd.read_csv("./metadata/val.csv")
    test_df_orig = pd.read_csv("./metadata/test.csv")

    # Create a balanced mini training set (ensure both classes are present for StratifiedKFold)
    # Select 10 positives and 40 negatives
    train_pos = train_df_orig[train_df_orig["label"] == 1].head(10)
    train_neg = train_df_orig[train_df_orig["label"] == 0].head(40)
    mini_train = (
        pd.concat([train_pos, train_neg])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Mini val set
    mini_val = val_df_orig.head(20)

    # Mini test set
    mini_test = test_df_orig.head(20)

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "train_mini.csv")
    mini_val_path = os.path.join(DEMO_DIR, "val_mini.csv")
    mini_test_path = os.path.join(DEMO_DIR, "test_mini.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Update Config to point to these new files
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path

    # Update Cache paths to avoid conflicts with real training artifacts
    Config.CACHE_TRAIN_DATA = os.path.join(DEMO_DIR, "train_waveforms_debug.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_DIR, "train_labels_debug.npy")
    Config.CACHE_VAL_DATA = os.path.join(DEMO_DIR, "val_waveforms_debug.npy")
    Config.CACHE_VAL_LABELS = os.path.join(DEMO_DIR, "val_labels_debug.npy")
    Config.CACHE_TEST_DATA = os.path.join(DEMO_DIR, "test_waveforms_debug.npy")
    Config.CACHE_TEST_CLIPS = os.path.join(DEMO_DIR, "test_clips_debug.npy")

    print(f"    Mini-train size: {len(mini_train)}")
    print(f"    Mini-test size: {len(mini_test)}")

    # 3. Data Loading
    print("[3] Testing Data Loading (library.data)...")
    # load_cached_data=False forces processing of our new mini files
    train_loader, valid_loader, test_loader, test_clips = get_data_loaders(
        fold=0, load_cached_data=False
    )

    # Verify Train Loader
    try:
        data_batch, label_batch = next(iter(train_loader))
        print(f"    Train Batch Shape: {data_batch.shape}")
        print(f"    Label Batch Shape: {label_batch.shape}")

        # Assertions
        assert data_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
        assert data_batch.shape[1] == 1, "Channel dimension mismatch (should be 1)"
        assert data_batch.shape[2] == Config.N_MELS, "Mel frequency dimension mismatch"
        # Time dimension check: 2.0s * 2000Hz = 4000 samples.
        # MelSpec (n_fft=1024, hop=64) -> 4000/64 + 1 approx 63.
        assert data_batch.shape[3] > 0, "Time dimension is zero"

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 4. Model Instantiation
    print("[4] Testing Model Instantiation (library.models)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    # Use resnet34 as it's standard and relatively fast.
    # pretrained=False to avoid downloading weights during demo.
    model = get_model("resnet34", pretrained=False)
    model = model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 1, 128, 64).to(device)
        output = model(dummy_input)
        assert output.shape == (2, 1), f"Model output shape mismatch: {output.shape}"
    print("    Model forward pass successful.")

    # 5. Training Loop
    print("[5] Testing Training Loop (library.engine)...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    save_path = os.path.join(DEMO_DIR, "demo_model.pth")
    log_path = os.path.join(DEMO_DIR, "demo_log.csv")

    best_val_loss = fit_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.EPOCHS,
        save_path=save_path,
        log_path=log_path,
    )

    assert os.path.exists(save_path), "Model checkpoint file was not created."
    print(f"    Training complete. Best Val Loss: {best_val_loss:.4f}")

    # 6. Prediction
    print("[6] Testing Prediction (library.engine)...")
    predictions = predict(model, test_loader, device)

    assert len(predictions) == len(
        mini_test
    ), "Number of predictions does not match test set size"
    assert (
        predictions.min() >= 0 and predictions.max() <= 1
    ), "Predictions out of probability range [0, 1]"
    print(f"    Generated {len(predictions)} predictions.")

    # 7. Stacking / Meta-Learner
    print("[7] Testing Stacking Module (library.stacking)...")

    # Simulate OOF predictions (Features for Meta-Learner)
    # In a real run, this would be the concat of val predictions from K folds.
    # Here we simulate for the size of our mini_train (which acts as the labeled set for meta-learner)
    n_samples_meta = len(mini_train)
    n_base_models = 3

    # Random probabilities simulating 3 base models
    X_train_meta = np.random.rand(n_samples_meta, n_base_models)
    y_train_meta = mini_train["label"].values

    # Train Meta Learner
    meta_model_path = os.path.join(DEMO_DIR, "meta_model.pkl")
    meta_model = train_meta_learner(
        X_train_meta, y_train_meta, save_path=meta_model_path
    )

    assert os.path.exists(meta_model_path), "Meta-learner model file not saved."

    # Generate Meta Submission
    # Simulate test predictions from base models
    X_test_meta = np.random.rand(len(mini_test), n_base_models)
    submission_path = os.path.join(DEMO_DIR, "demo_submission.csv")

    generate_meta_submission(
        meta_model, X_test_meta, test_clips, output_path=submission_path
    )

    assert os.path.exists(submission_path), "Submission file not created."

    # Verify Submission Format
    sub_df = pd.read_csv(submission_path)
    print(f"    Submission shape: {sub_df.shape}")
    assert list(sub_df.columns) == [
        "clip",
        "probability",
    ], "Submission columns mismatch"
    assert len(sub_df) == len(mini_test), "Submission row count mismatch"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
