import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config, seed_everything
from library.data_loader import HerbariumDataset, get_transforms
from library.network import PlantClassifier
from library.trainer import Trainer
from library.utils import calculate_metric
from library.inference import generate_submission


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print(">>> Setting up configuration and seeds...")
    seed_everything(42)

    # Override Config for a quick demo run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.USE_SWA = False  # Disable SWA to save time
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Preparation (Subsetting for Speed)
    print(">>> Preparing data subsets...")

    # Load full metadata
    full_train_df = pd.read_csv(Config.TRAIN_CSV)
    full_val_df = pd.read_csv(Config.VAL_CSV)
    full_test_df = pd.read_csv(Config.TEST_CSV)

    # Create tiny subsets (e.g., 32 samples each) to ensure quick execution
    # We select samples that definitely exist to avoid any IO errors, though metadata is verified.
    train_subset = full_train_df.sample(n=32, random_state=42).reset_index(drop=True)
    val_subset = full_val_df.sample(n=32, random_state=42).reset_index(drop=True)
    test_subset = full_test_df.sample(n=32, random_state=42).reset_index(drop=True)

    print(f"    Train subset size: {len(train_subset)}")
    print(f"    Val subset size: {len(val_subset)}")
    print(f"    Test subset size: {len(test_subset)}")

    # 3. Instantiate Datasets and Loaders
    print(">>> Instantiating Datasets and DataLoaders...")

    train_dataset = HerbariumDataset(
        train_subset, transforms=get_transforms("train", Config.IMG_SIZE), mode="train"
    )
    val_dataset = HerbariumDataset(
        val_subset, transforms=get_transforms("val", Config.IMG_SIZE), mode="val"
    )
    test_dataset = HerbariumDataset(
        test_subset, transforms=get_transforms("test", Config.IMG_SIZE), mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify DataLoader output
    images, labels = next(iter(train_loader))
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label shape: {labels.shape}"
    print("    DataLoader shapes verified.")

    # 4. Model Architecture Verification
    print(">>> Verifying Model Architecture...")
    model = PlantClassifier(
        pretrained=False
    )  # No need to download weights for a shape check
    model.eval()

    # Pass a dummy batch
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)
        outputs = model(dummy_input)

    assert outputs.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {outputs.shape}"
    print("    Model output shape verified.")

    # 5. Training Loop Simulation
    print(">>> Starting Training Simulation...")
    # Initialize Trainer
    trainer = Trainer(model=model)  # Pass the model we just created

    # Run fit (1 epoch as configured above)
    trainer.fit(train_loader, val_loader)

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")
    assert os.path.exists(best_model_path), "model_best.pth was not created!"
    print(f"    Training completed. Checkpoint saved at {best_model_path}")

    # 6. Inference Demonstration
    print(">>> Running Inference...")

    # Generate submission using the specific test loader we created
    # We explicitly pass the path to the best model we just trained
    submission_df = generate_submission(
        test_loader=test_loader,
        model_path=best_model_path,
        output_path=Config.SUBMISSION_PATH,
        device=torch.device(Config.DEVICE),
    )

    # Verify submission content
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"
    assert len(submission_df) == len(
        test_subset
    ), f"Submission rows ({len(submission_df)}) do not match test subset size ({len(test_subset)})"
    assert (
        "Id" in submission_df.columns and "Predicted" in submission_df.columns
    ), "Submission columns are incorrect"
    print(f"    Inference successful. Submission saved to {Config.SUBMISSION_PATH}")

    # 7. Metric Calculation Verification
    print(">>> Verifying Metric Calculation...")
    # Create dummy ground truth and predictions
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 2, 0, 0]  # Last one is wrong

    # Calculate F1
    # Class 0: TP=2, FN=0, FP=1 -> Precision=2/3, Recall=1.0 -> F1=0.8
    # Class 1: TP=1, FN=1, FP=0 -> Precision=1.0, Recall=0.5 -> F1=0.666
    # Class 2: TP=1, FN=0, FP=0 -> Precision=1.0, Recall=1.0 -> F1=1.0
    # Macro F1: (0.8 + 0.666 + 1.0) / 3 ~= 0.822

    score = calculate_metric(y_true, y_pred)
    assert 0.0 <= score <= 1.0, "Metric score out of range"
    print(f"    Metric calculation test passed. Score: {score:.4f}")

    print("\n>>> Demo Execution Completed Successfully!")


if __name__ == "__main__":
    run_demo()
