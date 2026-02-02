import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# 1. Suppress Warnings and Progress Bars
warnings.filterwarnings("ignore")

# Monkey-patch tqdm to disable progress bars in the provided library files
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# 2. Import Library Modules
# We import after patching tqdm to ensure the library uses the silent version
from library.utils import seed_everything
from library.data import LungDataset, get_transforms
from library.model import NSHDAN
from library.train import train_model, generate_submission


def check_data_loading():
    print("\n[1] Verifying Data Loading...")

    # Initialize dataset with a small limit for speed
    # We use 'train' mode to check label availability
    dataset = LungDataset(
        mode="train",
        transform=get_transforms("train"),
        limit_size=10,
        cache_dir="./working/cache_test",
    )

    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty."

    # Fetch a single sample
    sample = dataset[0]

    # Verify keys
    expected_keys = [
        "image_axial",
        "image_coronal",
        "meta",
        "target",
        "week",
        "base_fvc",
        "patient_week",
    ]
    for key in expected_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Verify Shapes
    # Images should be (3, 224, 224) based on Tri-Slab generation and transforms
    img_ax = sample["image_axial"]
    img_cor = sample["image_coronal"]
    meta = sample["meta"]

    assert img_ax.shape == (
        3,
        224,
        224,
    ), f"Unexpected Axial Image shape: {img_ax.shape}"
    assert img_cor.shape == (
        3,
        224,
        224,
    ), f"Unexpected Coronal Image shape: {img_cor.shape}"

    # Meta should be (4,) -> [Age, Sex, Smoking, Percent]
    assert meta.shape == (4,), f"Unexpected Metadata shape: {meta.shape}"

    print("Data loading verification passed.")
    return sample


def check_model_forward_pass(sample):
    print("\n[2] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NSHDAN().to(device)
    model.eval()

    # Prepare batch of size 2
    img_ax = torch.stack([sample["image_axial"], sample["image_axial"]]).to(device)
    img_cor = torch.stack([sample["image_coronal"], sample["image_coronal"]]).to(device)
    meta = torch.stack([sample["meta"], sample["meta"]]).to(device)

    print(f"Input Axial Shape: {img_ax.shape}")
    print(f"Input Meta Shape: {meta.shape}")

    with torch.no_grad():
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, meta)

    # Verify output shapes
    # Expecting (Batch_Size,) for each output
    batch_size = img_ax.size(0)

    assert alpha.shape == (batch_size,), f"Alpha shape mismatch: {alpha.shape}"
    assert sigma_base.shape == (
        batch_size,
    ), f"Sigma Base shape mismatch: {sigma_base.shape}"
    assert sigma_growth.shape == (
        batch_size,
    ), f"Sigma Growth shape mismatch: {sigma_growth.shape}"

    # Verify positivity of sigmas (Softplus used in model)
    assert torch.all(sigma_base > 0), "Sigma base must be positive"
    assert torch.all(sigma_growth > 0), "Sigma growth must be positive"

    print("Model forward pass verification passed.")


def check_training_pipeline():
    print("\n[3] Verifying Training Pipeline...")

    # Run training for a very short duration
    # limit_size=32 ensures we have enough for a batch but runs quickly
    # epochs=2 to test the loop transition
    best_loss = train_model(epochs=2, batch_size=4, lr=1e-3, patience=2, limit_size=32)

    print(f"Training finished with Best Val Loss: {best_loss}")

    # Verify model artifact creation
    model_path = "./working/best_model.pth"
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"

    print("Training pipeline verification passed.")


def check_submission_generation():
    print("\n[4] Verifying Submission Generation...")

    # Generate submission using the model trained in the previous step
    generate_submission(batch_size=8)

    sub_path = "./submission/submission.csv"
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    # Verify content format
    df = pd.read_csv(sub_path)
    print(f"Submission rows: {len(df)}")

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in df.columns, f"Missing column in submission: {col}"

    # Verify Confidence clipping logic (min 70)
    min_conf = df["Confidence"].min()
    assert min_conf >= 70, f"Confidence values found below 70: {min_conf}"

    print("Submission generation verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # Execute checks
    try:
        sample_data = check_data_loading()
        check_model_forward_pass(sample_data)
        check_training_pipeline()
        check_submission_generation()

        print("\n" + "=" * 40)
        print("ALL CHECKS PASSED SUCCESSFULLY")
        print("=" * 40)

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        # Print traceback for debugging if needed, but keeping it clean for now
        import traceback

        traceback.print_exc()
        sys.exit(1)
