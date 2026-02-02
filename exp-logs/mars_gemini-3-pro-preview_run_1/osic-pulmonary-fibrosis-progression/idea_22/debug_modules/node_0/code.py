import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import prepare_data, LungDataset, get_transforms
from library.model import CASDAN, LaplaceLogLikelihoodLoss
from library.train import train_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_metadata(source_csv, target_csv, n_patients=2, is_test=False):
    """
    Creates a small subset of the metadata CSV to speed up data loading.
    Selects all rows corresponding to the first `n_patients`.
    """
    df = pd.read_csv(source_csv)

    if is_test:
        # Test CSV has Patient_Week, need to extract Patient
        # Schema: Patient_Week, FVC, Confidence, Patient, ...
        patients = df["Patient"].unique()[:n_patients]
        subset_df = df[df["Patient"].isin(patients)].copy()
    else:
        # Train/Val CSV has Patient column
        patients = df["Patient"].unique()[:n_patients]
        subset_df = df[df["Patient"].isin(patients)].copy()

    subset_df.to_csv(target_csv, index=False)
    print(
        f"Created subset {target_csv} with {len(subset_df)} rows ({n_patients} patients)."
    )
    return subset_df


def verify_metric():
    """
    Verifies the get_score function against manual calculation.
    Formula: - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    """
    print("\n=== Verifying Metric (Laplace Log Likelihood) ===")
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([100])  # > 70, so not clipped

    # Manual Calc
    # Delta = 0
    # Sigma_clipped = 100
    # Term 1 = 0
    # Term 2 = ln(sqrt(2) * 100) = ln(141.421356) ~= 4.95174
    # Metric = -4.95174

    score = get_score(y_true, y_pred, sigma)
    expected = -np.log(np.sqrt(2) * 100)

    print(f"Calculated Score: {score:.5f}")
    print(f"Expected Score:   {expected:.5f}")

    assert np.isclose(score, expected, atol=1e-4), "Metric calculation mismatch!"
    print("Metric verification passed.")


def verify_data_pipeline(device):
    """
    Verifies data loading, transforms, and dataset item structure.
    """
    print("\n=== Verifying Data Pipeline ===")

    # Load the subsetted training data
    # prepare_data handles caching. We rely on the subset CSVs pointed to by Config.
    train_dataset = prepare_data("train", load_cached_data=False)

    print(f"Dataset length: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Dataset is empty!"

    # Fetch one item
    item = train_dataset[0]

    # Check keys
    expected_keys = {"axial", "coronal", "tabular", "target", "week", "patient"}
    assert expected_keys.issubset(
        item.keys()
    ), f"Missing keys in dataset item. Found: {item.keys()}"

    # Check Shapes
    # Image: (3, 224, 224) - Albumentations ToTensorV2 produces (C, H, W)
    # Note: generate_tri_slab produces (224, 224, 3), ToTensorV2 converts to (3, 224, 224)
    assert item["axial"].shape == (
        3,
        224,
        224,
    ), f"Axial shape mismatch: {item['axial'].shape}"
    assert item["coronal"].shape == (
        3,
        224,
        224,
    ), f"Coronal shape mismatch: {item['coronal'].shape}"

    # Tabular: 7 features (Age, Sex, Smk(3), Pct, BaseFVC)
    assert item["tabular"].shape == (
        7,
    ), f"Tabular shape mismatch: {item['tabular'].shape}"

    # Target and Week should be scalars (0-d tensors)
    assert item["target"].ndim == 0, "Target should be scalar"
    assert item["week"].ndim == 0, "Week should be scalar"

    print("Data pipeline verification passed.")
    return train_dataset


def verify_model_logic(device):
    """
    Verifies model instantiation, forward pass, and loss calculation.
    """
    print("\n=== Verifying Model Logic ===")

    model = CASDAN().to(device)
    criterion = LaplaceLogLikelihoodLoss()

    # Create dummy batch (Batch Size = 2)
    B = 2
    dummy_ax = torch.randn(B, 3, 224, 224).to(device)
    dummy_cor = torch.randn(B, 3, 224, 224).to(device)
    dummy_tab = torch.randn(B, 7).to(device)
    dummy_target = torch.tensor([2000.0, 2500.0]).to(device)
    dummy_weeks = torch.tensor([10.0, 12.0]).to(device)

    # Forward Pass
    alpha, sigma_base, sigma_growth = model(dummy_ax, dummy_cor, dummy_tab)

    print(
        f"Model Outputs - Alpha: {alpha.shape}, SigmaBase: {sigma_base.shape}, SigmaGrowth: {sigma_growth.shape}"
    )

    # Assert Output Shapes
    assert alpha.shape == (B,), "Alpha shape mismatch"
    assert sigma_base.shape == (B,), "Sigma Base shape mismatch"
    assert sigma_growth.shape == (B,), "Sigma Growth shape mismatch"

    # Assert Positivity of Sigmas (Softplus used in model)
    assert (sigma_base > 0).all(), "Sigma base must be positive"
    assert (sigma_growth > 0).all(), "Sigma growth must be positive"

    # Calculate Parametric Predictions manually for Loss check
    # Reconstruct Base FVC (dummy logic matching model assumption)
    # In model training loop: base_fvc_rec = tab[:, 6] * 1000.0 + 2500.0
    base_fvc_rec = dummy_tab[:, 6] * 1000.0 + 2500.0
    fvc_pred = base_fvc_rec + alpha * dummy_weeks
    sigma_pred = sigma_base + sigma_growth * torch.abs(dummy_weeks)

    # Loss Calculation
    loss = criterion(fvc_pred, sigma_pred, dummy_target)

    print(f"Computed Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print("Model logic verification passed.")
    return model


def verify_training_loop(model, train_dataset, device):
    """
    Runs a minimal training loop (1 epoch) and validation.
    """
    print("\n=== Verifying Training Loop ===")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = LaplaceLogLikelihoodLoss()

    # Run 1 Epoch
    loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Epoch Loss: {loss:.4f}")

    # Run Validation (using same dataset for demo purposes)
    val_score = validate(model, train_loader, device)
    print(f"Validation Score: {val_score:.4f}")

    # Save checkpoint for inference test
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not saved"

    print("Training loop verification passed.")


def verify_inference(device):
    """
    Verifies the inference pipeline using the saved model and test subset.
    """
    print("\n=== Verifying Inference Pipeline ===")

    # Prepare Test Data
    test_dataset = prepare_data("test", load_cached_data=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Load Model
    model = CASDAN().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            ax = batch["axial"].to(device)
            cor = batch["coronal"].to(device)
            tab = batch["tabular"].to(device)

            # Meta data extraction
            base_fvc = batch["meta"]["Baseline_FVC"].to(device)
            base_week = batch["meta"]["Baseline_Week"].to(device)
            pred_week = batch["meta"]["Predict_Week"].to(device)
            patient_weeks = batch["meta"]["Patient_Week"]

            alpha, sigma_base, sigma_growth = model(ax, cor, tab)

            delta_t = pred_week - base_week
            fvc_pred = base_fvc + alpha * delta_t
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_t)
            sigma_pred = torch.clamp(sigma_pred, min=Config.CONFIDENCE_CLIP)

            fvc_np = fvc_pred.cpu().numpy()
            sigma_np = sigma_pred.cpu().numpy()

            for i in range(len(patient_weeks)):
                results.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": fvc_np[i],
                        "Confidence": sigma_np[i],
                    }
                )

    # Save Submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Generated predictions for {len(sub_df)} rows.")
    print(sub_df.head())

    # Verify Submission Format
    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns
    assert len(sub_df) == len(test_dataset), "Submission row count mismatch"

    print("Inference verification passed.")


if __name__ == "__main__":
    # 1. Setup Environment and Config
    seed_everything(42)
    device = torch.device(Config.DEVICE)

    # Patch Config for Demo/Speed
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(
        Config.WORKING_DIR, "checkpoints", "best_model.pth"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    # 2. Create Data Subsets
    # We point Config to these new subset files so prepare_data reads them instead of full metadata
    Config.TRAIN_CSV = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    Config.VAL_CSV = os.path.join(Config.WORKING_DIR, "val_subset.csv")
    Config.TEST_CSV = os.path.join(Config.WORKING_DIR, "test_subset.csv")

    # Use original metadata paths to create subsets
    orig_train_csv = "./metadata/train.csv"
    orig_val_csv = "./metadata/val.csv"
    orig_test_csv = "./metadata/test.csv"

    create_subset_metadata(orig_train_csv, Config.TRAIN_CSV, n_patients=2)
    create_subset_metadata(orig_val_csv, Config.VAL_CSV, n_patients=1)
    create_subset_metadata(orig_test_csv, Config.TEST_CSV, n_patients=1, is_test=True)

    # 3. Execution Steps
    try:
        verify_metric()
        train_ds = verify_data_pipeline(device)
        model = verify_model_logic(device)
        verify_training_loop(model, train_ds, device)
        verify_inference(device)

        print("\n" + "=" * 40)
        print("ALL DEMONSTRATION STEPS COMPLETED SUCCESSFULLY")
        print("=" * 40)

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        raise e
