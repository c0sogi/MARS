import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import set_seed
from library.dataset import load_dataframe, BirdDataset, get_transforms
from library.model import BirdClassifier
from library.train import run_training
from library.predict import generate_predictions


def main():
    print("=== Bird Species Classification Library Demo ===\n")

    # 1. Configuration Setup for Speed
    # We override defaults to ensure the demo runs quickly (Debug mode, few epochs, small batch)
    print("[1/5] Setting up configuration...")

    # Define a separate working directory for this demo
    DEMO_WORKING_DIR = "./working/demo_run"
    DEMO_MODEL_PATH = os.path.join(DEMO_WORKING_DIR, "demo_model.pth")
    DEMO_SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "demo_submission.csv")

    # Update Config class attributes
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.MODEL_SAVE_PATH = DEMO_MODEL_PATH
    Config.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
    Config.setup()  # Create directories

    # Set reproducible seed
    set_seed(Config.SEED)

    # Runtime parameters for fast execution
    DEBUG_MODE = True
    BATCH_SIZE = 4
    EPOCHS = 2
    NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Update Config for workers as it is used inside DataLoader init in the library
    Config.NUM_WORKERS = NUM_WORKERS

    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Debug Mode: {DEBUG_MODE}")
    print("   Configuration setup complete.")

    # 2. Dataset and Transform Verification
    print("\n[2/5] Verifying Dataset and Transforms...")

    # Load training dataframe (subset due to debug=True)
    df_train = load_dataframe(Config.TRAIN_CSV, debug=DEBUG_MODE)
    print(f"   Loaded {len(df_train)} training samples (Debug Subset).")

    # Initialize Dataset
    transforms = get_transforms("train")
    dataset = BirdDataset(df_train, transforms=transforms)

    # Fetch a single sample to verify pipeline
    image, labels, rec_id = dataset[0]

    # Assertions
    assert isinstance(image, torch.Tensor), "Output image is not a torch.Tensor"
    # Check shape: (Channels, Height, Width) -> (3, 224, 224)
    expected_shape = (3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        image.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {image.shape}"
    # Check labels: (Num_Classes,) -> (19,)
    assert labels.shape == (
        Config.NUM_CLASSES,
    ), f"Label shape mismatch. Expected ({Config.NUM_CLASSES},), got {labels.shape}"
    assert isinstance(
        rec_id, (int, np.integer)
    ), f"rec_id should be integer, got {type(rec_id)}"

    print("   Dataset verification passed: Shapes and types are correct.")

    # 3. Model Architecture Verification
    print("\n[3/5] Verifying Model Architecture...")

    device = Config.DEVICE
    model = BirdClassifier(
        model_name="resnet18",  # Use lightweight model for demo
        pretrained=False,  # Skip downloading weights for speed/offline safety
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    model.eval()

    # Create a dummy batch
    dummy_input = torch.randn(BATCH_SIZE, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1]).to(
        device
    )

    with torch.no_grad():
        logits = model(dummy_input)

    # Verify output shape: (Batch_Size, Num_Classes)
    assert logits.shape == (
        BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    print("   Model verification passed: Forward pass successful.")

    # 4. Training Loop Demonstration
    print("\n[4/5] Running Training Loop...")

    # Run training using the library function
    # We explicitly pass parameters to override defaults in the function signature
    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        debug=DEBUG_MODE,
        save_path=DEMO_MODEL_PATH,
    )

    # Verify that the model checkpoint was saved
    assert os.path.exists(
        DEMO_MODEL_PATH
    ), f"Model checkpoint not found at {DEMO_MODEL_PATH}"
    print("   Training demonstration passed: Model checkpoint saved.")

    # 5. Inference Demonstration
    print("\n[5/5] Running Inference...")

    # Generate predictions using the library function
    generate_predictions(
        model_path=DEMO_MODEL_PATH,
        output_path=DEMO_SUBMISSION_PATH,
        batch_size=BATCH_SIZE,
        debug=DEBUG_MODE,
        device=device,
    )

    # Verify submission file
    assert os.path.exists(
        DEMO_SUBMISSION_PATH
    ), f"Submission file not found at {DEMO_SUBMISSION_PATH}"

    # Verify submission content format
    df_submission = pd.read_csv(DEMO_SUBMISSION_PATH)
    assert "Id" in df_submission.columns, "Submission missing 'Id' column"
    assert (
        "Probability" in df_submission.columns
    ), "Submission missing 'Probability' column"
    assert len(df_submission) > 0, "Submission file is empty"

    # Check if probabilities are valid (0-1) - though sigmoid ensures this, we verify the file content
    probs = df_submission["Probability"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print(
        f"   Inference demonstration passed: Generated {len(df_submission)} predictions."
    )
    print("\n=== All System Checks Passed Successfully ===")


if __name__ == "__main__":
    main()
