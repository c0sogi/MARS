import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, MixupHandler
from library.data import get_dataloaders, get_test_dataloaders
from library.models import get_cnn_model, SymbolicMLP
from library.training import Trainer
from library.inference import EnsemblePredictor


def run_demo():
    # =========================================================================
    # 1. Configuration Setup for Demo
    # =========================================================================
    print("Step 1: Setting up configuration...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce computational load
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.CNN_MODELS = ["resnet18"]  # Only test one model type
    Config.N_FOLDS = (
        2  # Only iterate 2 folds for demo logic (though we might only run 1)
    )

    # Setup directories
    Config.setup()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    seed_everything(Config.SEED)

    # =========================================================================
    # 2. Data Loading Verification
    # =========================================================================
    print("\nStep 2: Verifying Data Loading...")

    # Get dataloaders for Fold 0
    # load_cached_data=False forces fresh loading to test logic
    dataloaders = get_dataloaders(fold_idx=0, load_cached_data=False)

    # Check keys
    assert "train_cnn" in dataloaders
    assert "train_mlp" in dataloaders

    # Test CNN Batch
    cnn_batch = next(iter(dataloaders["train_cnn"]))
    imgs = cnn_batch["image"]
    labels = cnn_batch["labels"]
    rec_ids = cnn_batch["rec_id"]

    # Expected shape: (Batch, 3, 224, 224)
    assert imgs.dim() == 4
    assert imgs.shape[1] == 3
    assert imgs.shape[2] == 224
    assert imgs.shape[3] == 224
    assert labels.shape[1] == Config.NUM_CLASSES
    print(f"CNN Batch Verified: {imgs.shape}")

    # Test MLP Batch
    mlp_batch = next(iter(dataloaders["train_mlp"]))
    feats = mlp_batch["features"]
    mlp_labels = mlp_batch["labels"]

    # Expected shape: (Batch, 100)
    assert feats.dim() == 2
    assert feats.shape[1] == Config.MLP_INPUT_DIM
    assert mlp_labels.shape[1] == Config.NUM_CLASSES
    print(f"MLP Batch Verified: {feats.shape}")

    # =========================================================================
    # 3. Model Verification
    # =========================================================================
    print("\nStep 3: Verifying Model Architectures...")

    # Test CNN Model
    cnn_model = get_cnn_model("resnet18", pretrained=False).to(device)
    cnn_out = cnn_model(imgs.to(device))
    assert cnn_out.shape == (imgs.shape[0], Config.NUM_CLASSES)
    print("CNN Model Forward Pass Successful.")

    # Test MLP Model
    mlp_model = SymbolicMLP().to(device)
    mlp_out = mlp_model(feats.to(device))
    assert mlp_out.shape == (feats.shape[0], Config.NUM_CLASSES)
    print("MLP Model Forward Pass Successful.")

    # =========================================================================
    # 4. Training Loop Verification
    # =========================================================================
    print("\nStep 4: Verifying Training Loop...")

    # Train MLP (Fast)
    print("Training MLP for 2 epochs...")
    mlp_trainer = Trainer(
        model=mlp_model,
        train_loader=dataloaders["train_mlp"],
        val_loader=dataloaders["val_mlp"],
        device=device,
        model_name="mlp",
        fold_idx=0,
        is_mlp=True,
    )
    mlp_trainer.fit(epochs=Config.EPOCHS)

    # Verify Checkpoint Creation
    mlp_checkpoints = os.listdir(os.path.join(Config.CHECKPOINT_DIR, "mlp"))
    assert len(mlp_checkpoints) > 0, "MLP checkpoints were not saved."
    print(f"MLP Checkpoints created: {len(mlp_checkpoints)}")

    # Train CNN (Short run)
    print("Training CNN (ResNet18) for 1 epoch...")
    cnn_trainer = Trainer(
        model=cnn_model,
        train_loader=dataloaders["train_cnn"],
        val_loader=dataloaders["val_cnn"],
        device=device,
        model_name="resnet18",
        fold_idx=0,
        is_mlp=False,
    )
    cnn_trainer.fit(epochs=1)  # Just 1 epoch to save time

    # Verify Checkpoint Creation
    cnn_checkpoints = os.listdir(os.path.join(Config.CHECKPOINT_DIR, "resnet18"))
    assert len(cnn_checkpoints) > 0, "CNN checkpoints were not saved."
    print(f"CNN Checkpoints created: {len(cnn_checkpoints)}")

    # =========================================================================
    # 5. Inference Verification
    # =========================================================================
    print("\nStep 5: Verifying Inference and Submission...")

    # Initialize Predictor
    # It should pick up the checkpoints we just created
    predictor = EnsemblePredictor(load_cached_data=False)

    # Run Prediction
    predictor.predict()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check format
    assert "Id" in sub_df.columns
    assert "Probability" in sub_df.columns
    assert not sub_df.isnull().values.any()

    # Check Id mapping logic (should be rec_id * 100 + species)
    # Test set has 64 samples * 19 species = 1216 rows
    # Note: The test set size comes from metadata/test.csv (64 samples)
    expected_rows = 64 * 19
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # =========================================================================
    # 6. Utils Verification
    # =========================================================================
    print("\nStep 6: Verifying Utilities...")

    # Test ROC AUC
    y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]])
    score = calculate_roc_auc(y_true, y_pred)
    assert score == 1.0, f"Expected AUC 1.0, got {score}"

    # Test Mixup
    mixup = MixupHandler(alpha=1.0)
    x = torch.ones((4, 10))
    y = torch.ones((4, 2))
    mixed_x, mixed_y = mixup.apply(x, y)
    assert mixed_x.shape == x.shape
    assert mixed_y.shape == y.shape
    print("Utilities Verified.")

    print("\nAll verification steps passed successfully!")


if __name__ == "__main__":
    run_demo()
