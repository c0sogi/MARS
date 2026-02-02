import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import library components
from library.config import Config, seed_everything
from library.data import LungDataset, get_transforms
from library.model import TSBCNet, laplace_log_likelihood_loss
from library.utils import calculate_metric
from library.train import run


def demo_components():
    """
    Demonstrates instantiation and basic usage of Data, Model, and Loss components.
    """
    print("--- Demonstrating Component Usage ---")

    # 1. Setup Configuration and Seeds
    Config.setup()
    seed_everything(42)
    device = torch.device("cpu")  # Use CPU for simple shape assertions

    # 2. Data Loading Verification
    print("1. Verifying Dataset...")
    # Load a tiny subset of metadata for verification purposes
    # We rely on the metadata generated in previous steps
    try:
        df = pd.read_csv(Config.TRAIN_CSV).head(10)
    except FileNotFoundError:
        print("Metadata not found. Ensure metadata generation script has run.")
        return

    # Instantiate Dataset in 'train' mode
    # This triggers the cache generation for these 10 patients if not present
    ds = LungDataset(
        df, mode="train", transform=get_transforms("train"), load_cached_data=True
    )

    # Fetch one sample to verify structure
    sample = ds[0]

    # Assertions for Data Shapes
    # Image shape: (Channels, H, W). Config.IMG_SIZE is 224. TriSlab produces 3 channels.
    img_ax = sample["image_axial"]
    img_cor = sample["image_coronal"]
    tabular = sample["tabular"]
    target = sample["target"]

    assert img_ax.shape == (3, 224, 224), f"Axial image shape mismatch: {img_ax.shape}"
    assert img_cor.shape == (
        3,
        224,
        224,
    ), f"Coronal image shape mismatch: {img_cor.shape}"
    assert tabular.shape == (6,), f"Tabular feature shape mismatch: {tabular.shape}"
    assert isinstance(target, torch.Tensor), "Target should be a torch.Tensor"

    print("   Dataset verification passed. Shapes are correct.")

    # 3. Model Logic Verification
    print("2. Verifying Model Forward Pass...")
    model = TSBCNet().to(device)
    model.eval()

    # Create dummy batch of size 2
    B = 2
    dummy_ax = torch.randn(B, 3, 224, 224).to(device)
    dummy_cor = torch.randn(B, 3, 224, 224).to(device)
    dummy_tab = torch.randn(B, 6).to(device)

    with torch.no_grad():
        # Forward pass returns [alpha, sigma_base, sigma_growth]
        output = model(dummy_ax, dummy_cor, dummy_tab)

    # Output should be (Batch_Size, 3)
    assert output.shape == (B, 3), f"Model output shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("   Model verification passed. Output shape is (B, 3).")

    # 4. Loss Function Verification
    print("3. Verifying Loss Function...")
    # Dummy targets (FVC)
    y_true = torch.tensor([2000.0, 2500.0]).to(device)

    # Dummy predictions (FVC_pred, Sigma_pred)
    y_pred = torch.tensor([2100.0, 2400.0]).to(device)
    sigma = torch.tensor([100.0, 150.0]).to(device)

    loss = laplace_log_likelihood_loss(y_true, y_pred, sigma)

    # Loss should be a scalar tensor
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    print("   Loss verification passed.")


def run_pipeline_demo():
    """
    Runs the full training pipeline using the `run` wrapper from library.train.
    Uses debug mode for speed.
    """
    print("\n--- Running Full Training Pipeline (Debug Mode) ---")

    # We use debug=True to limit data size to Config.DEBUG_SAMPLES (50)
    # We set epochs=1 and batch_size=4 to ensure the run finishes in seconds
    try:
        run(debug=True, epochs=1, batch_size=4)
        print("Pipeline execution completed successfully.")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise e

    # Verify that the submission file was generated
    sub_path = "./submission/submission.csv"
    if os.path.exists(sub_path):
        df_sub = pd.read_csv(sub_path)
        print(f"Submission generated at {sub_path}")
        print(f"Rows: {len(df_sub)}")

        # Check required columns
        required_cols = ["Patient_Week", "FVC", "Confidence"]
        for col in required_cols:
            assert col in df_sub.columns, f"Missing column in submission: {col}"

        print("Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # 1. Verify individual components (Data, Model, Loss)
    demo_components()

    # 2. Run the full training/inference pipeline on a subset
    run_pipeline_demo()
