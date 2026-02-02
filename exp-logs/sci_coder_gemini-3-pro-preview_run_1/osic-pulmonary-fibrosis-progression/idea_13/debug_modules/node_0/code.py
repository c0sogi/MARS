import os
import sys
import shutil
import pandas as pd
import torch
import torch.optim as optim
import warnings

# Import provided library components
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DualAxisNet
from library.trainer import LaplaceLoss, train_one_epoch, validate, predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_metadata(
    source_dir, dest_dir, n_patients_train=3, n_patients_val=2, n_patients_test=2
):
    """
    Creates a small subset of the metadata files to speed up the demonstration.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Process Train
    train_df = pd.read_csv(os.path.join(source_dir, "train.csv"))
    train_patients = train_df["Patient"].unique()[:n_patients_train]
    train_subset = train_df[train_df["Patient"].isin(train_patients)]
    train_subset.to_csv(os.path.join(dest_dir, "train.csv"), index=False)

    # Process Val
    val_df = pd.read_csv(os.path.join(source_dir, "val.csv"))
    val_patients = val_df["Patient"].unique()[:n_patients_val]
    val_subset = val_df[val_df["Patient"].isin(val_patients)]
    val_subset.to_csv(os.path.join(dest_dir, "val.csv"), index=False)

    # Process Test
    test_df = pd.read_csv(os.path.join(source_dir, "test.csv"))
    test_patients = test_df["Patient"].unique()[:n_patients_test]
    test_subset = test_df[test_df["Patient"].isin(test_patients)]
    test_subset.to_csv(os.path.join(dest_dir, "test.csv"), index=False)

    print(f"Created metadata subset in {dest_dir}")
    print(f"  Train: {len(train_subset)} rows ({len(train_patients)} patients)")
    print(f"  Val:   {len(val_subset)} rows ({len(val_patients)} patients)")
    print(f"  Test:  {len(test_subset)} rows ({len(test_patients)} patients)")


def main():
    print("Initializing Lung Function Prediction Demo...")

    # 1. Configuration and Setup
    seed_everything(42)

    # Define paths
    BASE_DIR = "./working/demo_run"
    META_DIR = os.path.join(BASE_DIR, "metadata")
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

    # Clean up previous run if exists
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    # 2. Prepare Data Subset
    # We use the existing ./metadata as source and create a subset in ./working
    create_subset_metadata("./metadata", META_DIR)

    # 3. Initialize Data Loaders
    # We use num_workers=0 to avoid multiprocessing overhead for this small demo
    print("\nLoading Datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_dir=META_DIR,
        cache_dir=CACHE_DIR,
        batch_size=4,
        num_workers=0,
        img_size=224,
    )

    # Verify Data Loading
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")

    # Assertions to ensure data pipeline is working
    assert "img_ax" in batch and "img_cor" in batch
    assert batch["img_ax"].shape == (
        4,
        3,
        224,
        224,
    ), f"Unexpected shape: {batch['img_ax'].shape}"
    assert "tab" in batch
    assert "meta" in batch
    assert "target" in batch

    # 4. Initialize Model
    # Determine tabular input dimension dynamically from the data
    tabular_dim = batch["tab"].shape[1]
    print(f"Detected Tabular Feature Dimension: {tabular_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # We use pretrained=False to avoid downloading weights during the demo
    model = DualAxisNet(tabular_input_dim=tabular_dim, embed_dim=128, pretrained=False)
    model.to(device)

    # 5. Setup Training Components
    criterion = LaplaceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # 6. Run Training Loop (Demo: 2 Epochs)
    print("\nStarting Training Loop...")
    epochs = 2
    for epoch in range(epochs):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

        # Assertions to ensure model is learning (producing valid scalar outputs)
        assert not pd.isna(train_loss), "Training loss is NaN"
        assert not pd.isna(val_score), "Validation score is NaN"

    # 7. Inference / Prediction
    print("\nGenerating Predictions...")
    predict(model, test_loader, device, output_path=SUBMISSION_PATH)

    # 8. Verify Submission
    if os.path.exists(SUBMISSION_PATH):
        sub_df = pd.read_csv(SUBMISSION_PATH)
        print("\nSubmission File Generated:")
        print(sub_df.head())

        # Validation checks on submission format
        assert "Patient_Week" in sub_df.columns
        assert "FVC" in sub_df.columns
        assert "Confidence" in sub_df.columns
        assert len(sub_df) > 0
        assert (
            sub_df["Confidence"].min() >= 70
        ), "Confidence values must be clipped at 70"

        print("\nDemo Completed Successfully!")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    main()
