import os
import torch
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import setup_reproducibility, collate_fn
from library.dataset import get_datasets, get_test_dataset
from library.model import get_model
from library.engine import train_one_epoch, evaluate, generate_submission


def main():
    # 1. Setup Environment and Reproducibility
    print("=== Setting up Environment ===")
    setup_reproducibility(Config.SEED)
    warnings.filterwarnings("ignore")

    # Check device
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Override Configuration for Fast Demonstration
    print("\n=== Configuring for Demo Speed ===")
    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 images for training/val/test

    # Reduce training parameters
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 3. Data Loading & Verification
    print("\n=== Initializing Datasets ===")
    # load_cached_data=False forces metadata processing to ensure our Debug settings apply
    # regardless of any pre-existing cache files
    train_dataset, val_dataset = get_datasets(load_cached_data=False)

    # Verify Dataset Length
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_dataset)}"
    assert (
        len(val_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Val dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(val_dataset)}"

    # Verify Data Item Structure
    sample_img, sample_target, sample_id = train_dataset[0]

    assert isinstance(sample_img, torch.Tensor), "Image is not a Tensor"
    assert sample_img.ndim == 3, "Image tensor should be 3-dimensional (C, H, W)"
    assert isinstance(sample_target, dict), "Target should be a dictionary"
    assert "boxes" in sample_target, "Target missing 'boxes'"
    assert "labels" in sample_target, "Target missing 'labels'"
    assert "study_label" in sample_target, "Target missing 'study_label'"

    print("Dataset structure verified successfully.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 4. Model Initialization & Verification
    print("\n=== Initializing Model ===")
    model = get_model()
    model.to(device)

    # Verify Forward Pass (Training Mode)
    print("Verifying model forward pass...")
    model.train()

    # Construct a dummy batch on the device
    dummy_imgs = [sample_img.to(device)]
    dummy_targets = [{k: v.to(device) for k, v in sample_target.items()}]

    # Forward pass returns a loss dictionary in training mode
    loss_dict = model(dummy_imgs, dummy_targets)

    assert isinstance(loss_dict, dict), "Model output should be a dict in training mode"
    assert "loss_study" in loss_dict, "Output missing 'loss_study' (Auxiliary Head)"
    assert "loss_classifier" in loss_dict, "Output missing 'loss_classifier' (ROI Head)"

    print("Model forward pass verified. Loss keys present.")

    # 5. Training Loop Demonstration
    print("\n=== Starting Training Demo (1 Epoch) ===")
    # Setup Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Run Training
    metrics = train_one_epoch(
        model, optimizer, train_loader, device, epoch=0, print_freq=2
    )

    print(f"Training finished. Avg Loss: {metrics['loss'].avg:.4f}")

    # 6. Evaluation Demonstration
    print("\n=== Starting Evaluation Demo ===")
    val_loss = evaluate(model, val_loader, device)
    print(f"Evaluation finished. Validation Loss: {val_loss:.4f}")

    # 7. Inference & Submission Demonstration
    print("\n=== Generating Submission Demo ===")
    # Initialize Test Dataset
    test_dataset = get_test_dataset(load_cached_data=False)

    # Verify Test Dataset respects debug size
    assert len(test_dataset) == Config.DEBUG_SAMPLE_SIZE

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Generate Submission
    generate_submission(model, test_loader, device)

    # Verify Output File
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file created at: {submission_path}")
        print(f"Rows: {len(df_sub)}")
        print("Columns:", df_sub.columns.tolist())

        # Basic content check
        assert "id" in df_sub.columns
        assert "PredictionString" in df_sub.columns
        assert not df_sub.empty
    else:
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
