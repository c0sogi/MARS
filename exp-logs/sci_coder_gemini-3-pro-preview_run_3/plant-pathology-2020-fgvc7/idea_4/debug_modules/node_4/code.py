import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything
from library.dataset import prepare_loaders, prepare_test_loader
from library.model import AppleClassifier
from library.train import run_training
from library.inference import generate_submission


def main():
    print("==== Starting Apple Disease Detection Demo ====")

    # 1. Configuration Override for Speed and Demo
    print("\n[1] Configuring environment...")
    CFG.debug = True
    CFG.debug_sample_size = 50  # Small subset for speed
    CFG.epochs = 1
    CFG.n_folds = 2  # Minimum 2 folds required for StratifiedKFold
    CFG.backbones = ["convnext_tiny.fb_in1k"]  # Use the smaller backbone
    CFG.batch_size = 4
    CFG.num_workers = 0  # Avoid multiprocessing overhead/issues in simple script
    CFG.output_dir = "./working/demo_output"
    CFG.submission_dir = "./working/demo_submission"

    # Ensure directories exist
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.submission_dir, exist_ok=True)

    # Set seed
    seed_everything(CFG.seed)
    print("    Configuration updated: Debug=True, Epochs=1, Backbone=convnext_tiny")

    # 2. Verify Data Loading
    print("\n[2] Verifying Data Loading...")
    train_loader, val_loader = prepare_loaders(
        fold=0, backbone="convnext_tiny.fb_in1k", debug=True
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.shape == (CFG.batch_size, 3, 224, 224), "Incorrect image batch shape"
    assert labels.shape == (CFG.batch_size, 4), "Incorrect label batch shape"
    assert images.dtype == torch.float32, "Images should be float32"
    print("    Data Loading verification passed.")

    # 3. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")
    model = AppleClassifier(model_name="convnext_tiny.fb_in1k", pretrained=False)
    model.to(CFG.device)
    model.eval()

    with torch.no_grad():
        # Move images to device
        images = images.to(CFG.device)
        output = model(images)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (CFG.batch_size, 4), "Model output shape mismatch"
    print("    Model verification passed.")

    # 4. Run Training (Fold 0)
    print("\n[4] Running Training for Fold 0...")
    # Note: run_training handles model instantiation internally with pretrained=True
    best_score = run_training(
        fold=0, backbone="convnext_tiny.fb_in1k", debug=True, epochs=1
    )

    print(f"    Training finished. Best ROC AUC: {best_score:.4f}")

    # Verify checkpoint existence
    expected_model_path = os.path.join(
        CFG.output_dir, "convnext_tiny.fb_in1k_fold0_best.pth"
    )
    if os.path.exists(expected_model_path):
        print(f"    Checkpoint saved successfully at: {expected_model_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {expected_model_path}")

    # 5. Run Inference and Generate Submission
    print("\n[5] Running Inference and Generating Submission...")
    # generate_submission loops through CFG.backbones and CFG.n_folds
    # It will look for the model we just trained
    generate_submission(load_cached_data=False)

    expected_sub_path = os.path.join(CFG.submission_dir, "submission.csv")

    if os.path.exists(expected_sub_path):
        print(f"    Submission file generated at: {expected_sub_path}")

        # Validate submission format
        sub_df = pd.read_csv(expected_sub_path)
        print("    Submission Head:")
        print(sub_df.head(3))

        assert "image_id" in sub_df.columns, "image_id column missing"
        for label in CFG.class_labels:
            assert label in sub_df.columns, f"Column {label} missing in submission"

        assert len(sub_df) > 0, "Submission file is empty"
        print("    Submission format verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
