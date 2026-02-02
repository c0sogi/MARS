import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything, calculate_class_weights
from library.data import (
    AppleDataset,
    get_transforms,
    load_full_train_data,
    load_test_data,
)
from library.modeling import get_model
from library.training import train_one_epoch, valid_one_epoch
from library.inference import predict_all_folds

if __name__ == "__main__":
    print("Starting Apple Disease Detection Library Demo...")

    # ==========================================
    # 1. Configuration Overrides for Speed/Demo
    # ==========================================
    print("\n[1] Configuring environment...")

    # Set paths for this demo run
    CFG.working_dir = "./working/demo_run"
    CFG.models_dir = os.path.join(CFG.working_dir, "models")
    CFG.submission_dir = CFG.working_dir
    CFG.submission_path = os.path.join(CFG.submission_dir, "submission.csv")

    # Ensure directories exist
    os.makedirs(CFG.models_dir, exist_ok=True)
    os.makedirs(CFG.submission_dir, exist_ok=True)

    # Override hyperparameters for fast execution
    CFG.debug = True
    CFG.epochs = 1
    CFG.batch_size = 4
    CFG.num_workers = 0  # Avoid multiprocessing overhead in demo
    CFG.model_architectures = ["resnet18"]  # Use a smaller model for demo
    CFG.n_folds = 1  # Only run one fold

    # Set seed
    seed_everything(CFG.seed)
    print("Configuration updated for demo execution.")

    # ==========================================
    # 2. Data Loading & Utils Verification
    # ==========================================
    print("\n[2] Verifying Data Loading and Utils...")

    # Load Training Data (Debug subset)
    df_train = load_full_train_data(debug=True)
    print(f"Loaded train data shape: {df_train.shape}")
    assert len(df_train) > 0, "Training data should not be empty."

    # Verify Class Weights Calculation
    weights = calculate_class_weights(df_train, CFG.target_cols, device="cpu")
    print(f"Class weights: {weights}")
    assert weights.shape == (CFG.num_classes,), "Weights tensor shape mismatch."
    assert not torch.isnan(weights).any(), "Weights contain NaNs."

    # Verify Dataset and Transforms
    train_dataset = AppleDataset(df_train, transform=get_transforms("train"))
    image, label = train_dataset[0]

    print(f"Sample image shape: {image.shape}")
    print(f"Sample label: {label}")

    # Assertions
    assert image.shape == (
        3,
        CFG.img_size,
        CFG.img_size,
    ), f"Image shape mismatch. Expected (3, {CFG.img_size}, {CFG.img_size})"
    assert label.shape == (CFG.num_classes,), "Label shape mismatch."
    assert isinstance(image, torch.Tensor), "Image is not a tensor."
    assert isinstance(label, torch.Tensor), "Label is not a tensor."

    # ==========================================
    # 3. Model & Training Loop Verification
    # ==========================================
    print("\n[3] Verifying Model and Training Loop...")

    device = torch.device(CFG.device)

    # Initialize Model (pretrained=False to avoid download time for demo)
    model = get_model(
        CFG.model_architectures[0], num_classes=CFG.num_classes, pretrained=False
    )
    model.to(device)

    # Setup Training Components
    train_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=0
    )

    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run One Epoch of Training
    print("Running training for 1 epoch...")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

    assert isinstance(train_loss, float), "Train loss should be a float."
    assert train_loss > 0, "Train loss should be positive."

    # Run Validation (reuse train loader for speed in demo)
    print("Running validation...")
    val_loss, val_auc = valid_one_epoch(model, train_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Save the model to simulate the end of a training fold
    # This is required for the inference step to work
    save_path = os.path.join(CFG.models_dir, f"{CFG.model_architectures[0]}_fold_0.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to: {save_path}")
    assert os.path.exists(save_path), "Model checkpoint file was not created."

    # ==========================================
    # 4. Inference & Submission Verification
    # ==========================================
    print("\n[4] Verifying Inference and Submission...")

    # Run the inference pipeline provided in the library
    # This will load the test data, load the model we just saved, and generate a submission
    predict_all_folds(debug=True)

    # Verify Submission File
    if os.path.exists(CFG.submission_path):
        submission_df = pd.read_csv(CFG.submission_path)
        print(f"Submission generated. Shape: {submission_df.shape}")
        print(submission_df.head())

        # Assertions
        expected_cols = ["image_id"] + CFG.target_cols
        assert (
            list(submission_df.columns) == expected_cols
        ), "Submission columns mismatch."
        assert len(submission_df) > 0, "Submission file is empty."

        # Check values are probabilities (0-1)
        prob_cols = submission_df[CFG.target_cols]
        assert (prob_cols.values >= 0).all() and (
            prob_cols.values <= 1
        ).all(), "Predictions are not valid probabilities."

        print("Submission verification successful.")
    else:
        raise FileNotFoundError(f"Submission file not found at {CFG.submission_path}")

    print("\nAll demonstrations and verifications passed successfully!")
