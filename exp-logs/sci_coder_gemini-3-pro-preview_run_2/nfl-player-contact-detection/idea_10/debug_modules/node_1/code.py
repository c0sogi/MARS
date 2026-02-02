import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import FeatureEngineer
from library.dataset import ContactDataset
from library.model import EFWideResNet
from library.trainer import Trainer


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("Initializing Configuration...")

    # Modify Config for fast demonstration
    Config.DEBUG = True  # Subsamples data to 5000 rows for speed
    Config.EPOCHS = 2  # limit epochs
    Config.BATCH_SIZE = 1024  # Batch size
    Config.NUM_RES_BLOCKS = 1  # Reduce model complexity for speed

    # Set up a specific working directory for this execution
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Manually update paths that were defined at import time
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler_debug.joblib")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. Feature Engineering
    # =========================================================================
    print("\n--- Feature Engineering ---")
    # Initialize FeatureEngineer
    fe = FeatureEngineer(debug=Config.DEBUG)

    # Ensure the instance uses our updated paths
    fe.scaler_path = Config.SCALER_PATH
    fe.label_encoder_path = os.path.join(
        Config.WORKING_DIR, "label_encoder_debug.joblib"
    )

    # Process Training Data
    print("Processing Training Data...")
    # load_cached_data=False ensures we generate the debug subset fresh
    X_train, X_cat_train, y_train, ids_train = fe.process_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        is_train=True,
        load_cached_data=False,
    )

    # Validation assertions
    assert len(X_train) == len(y_train)
    assert not np.isnan(X_train).any(), "NaNs found in training data"
    print(f"Train Data Shape: {X_train.shape}")

    # Process Validation Data
    print("Processing Validation Data...")
    X_val, X_cat_val, y_val, ids_val = fe.process_dataset(
        Config.VAL_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        is_train=False,
        load_cached_data=False,
    )
    print(f"Validation Data Shape: {X_val.shape}")

    # =========================================================================
    # 3. Dataset & DataLoader
    # =========================================================================
    print("\n--- Data Loading ---")
    train_ds = ContactDataset(X_train, X_cat_train, y_train)
    val_ds = ContactDataset(X_val, X_cat_val, y_val)

    # num_workers=0 ensures stability in this script execution
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify batch retrieval
    sample_cont, sample_cat, sample_y = next(iter(train_loader))
    print(f"Sample Batch - Continuous: {sample_cont.shape}, Label: {sample_y.shape}")

    # =========================================================================
    # 4. Model Initialization
    # =========================================================================
    print("\n--- Model Initialization ---")
    num_continuous = X_train.shape[1]

    model = EFWideResNet(
        num_continuous_features=num_continuous,
        embedding_config=Config.EMBEDDING_CONFIG,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # =========================================================================
    # 5. Training
    # =========================================================================
    print("\n--- Training Loop ---")
    trainer = Trainer(model, train_loader, val_loader, optimizer, device=Config.DEVICE)

    best_threshold = trainer.fit(
        epochs=Config.EPOCHS, patience=2, save_path=Config.MODEL_PATH
    )

    print(f"Training finished. Best Threshold: {best_threshold}")

    # =========================================================================
    # 6. Inference (Test Set)
    # =========================================================================
    print("\n--- Inference ---")
    # Process Test Data (using test tracking)
    X_test, X_cat_test, y_test, ids_test = fe.process_dataset(
        Config.TEST_METADATA_PATH,
        Config.TEST_TRACKING_PATH,
        is_train=False,
        load_cached_data=False,
    )

    # Create Test Loader
    test_ds = ContactDataset(X_test, X_cat_test, y_test)
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Reuse trainer's validation logic for prediction
    trainer.val_loader = test_loader

    print("Generating predictions...")
    _, probs, _ = trainer.validate()

    # Apply optimized threshold
    preds = (probs >= best_threshold).astype(int)

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    print("\n--- Saving Submission ---")
    submission = pd.DataFrame({"contact_id": ids_test, "contact": preds})

    # Save to working directory
    sub_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(f"Submission Head:\n{submission.head()}")

    # Final Verification
    assert os.path.exists(sub_path), "Submission file was not created."
    assert len(submission) == len(X_test), "Submission length mismatch."

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
