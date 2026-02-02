import os
import torch
import pandas as pd
import numpy as np
from library.utils import set_seed
from library.dataset import create_dataloaders
from library.model import PlantClassifier
from library.trainer import Trainer
from library.inference import generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Set Seed for Reproducibility
    # Ensures consistent results across runs
    set_seed(42)
    print("Seed set to 42.")

    # 2. Demonstrate Data Loading
    print("\n[Step 1] Creating DataLoaders (Debug Mode)")
    # We use debug=True to load a small subset of the data (2000 train, 500 val/test)
    # This allows the script to run quickly within the time limit.
    batch_size = 8
    train_loader, val_loader, test_loader = create_dataloaders(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=2,
        debug=True,
        img_size=260,
    )

    # Validate Train Loader Structure
    try:
        images, labels = next(iter(train_loader))
        print(f"  Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        # Verify shapes: (Batch, Channels, Height, Width)
        assert images.shape == (batch_size, 3, 260, 260), "Train image shape mismatch"
        assert labels.shape == (batch_size,), "Train label shape mismatch"
        assert labels.dtype == torch.long, "Labels must be LongTensor"
    except StopIteration:
        raise RuntimeError("Train loader is empty.")

    # Validate Test Loader Structure
    try:
        test_images, test_ids = next(iter(test_loader))
        print(f"  Test Batch  - Images: {test_images.shape}, IDs: {len(test_ids)}")

        assert test_images.shape == (
            batch_size,
            3,
            260,
            260,
        ), "Test image shape mismatch"
        assert len(test_ids) == batch_size, "Test ID count mismatch"
        # IDs should be strings representing the image_id
        assert isinstance(test_ids[0], str), "Test IDs should be strings"
    except StopIteration:
        raise RuntimeError("Test loader is empty.")

    print("  DataLoaders verified successfully.")

    # 3. Demonstrate Model Initialization
    print("\n[Step 2] Initializing PlantClassifier")
    # Using 15501 classes as per dataset specifications
    # pretrained=False is used here to speed up initialization for the demo
    model = PlantClassifier(num_classes=15501, pretrained=False)
    model.eval()

    # Forward pass check with the batch loaded earlier
    with torch.no_grad():
        outputs = model(images)

    print(f"  Model Output Shape: {outputs.shape}")
    # Output should be (Batch Size, Number of Classes)
    assert outputs.shape == (batch_size, 15501), "Model output shape mismatch"
    print("  Model initialized and forward pass verified.")

    # 4. Demonstrate Training Pipeline
    print("\n[Step 3] Running Trainer (1 Epoch)")
    # Configure trainer with minimal settings for speed
    config = {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "patience": 2,
        "use_mixup": False,  # Disable mixup for this small batch demo
    }

    trainer = Trainer(config)

    # Run training for 1 epoch on the debug dataset
    # This verifies the training loop, loss calculation, backprop, and validation logic
    trainer.fit(train_loader, val_loader, epochs=1)

    # Check if a model checkpoint was saved
    # Note: If F1 score is 0.0 (possible with random init), it might not save over the initial 0.0 best_f1
    model_path = "./working/idea_2/best_model.pth"
    if os.path.exists(model_path):
        print(f"  Best model saved at: {model_path}")
    else:
        print(
            "  Note: No model saved (likely due to low metric score on random initialization)."
        )

    # 5. Demonstrate Prediction/Inference via Trainer
    print("\n[Step 4] Generating Predictions using Trainer")
    # This uses the internal predict method which handles TTA and submission file generation
    trainer.predict(test_loader)

    submission_path = "./submission/submission.csv"
    assert os.path.exists(
        submission_path
    ), "Submission file was not created by Trainer.predict"

    sub_df = pd.read_csv(submission_path)
    print(f"  Submission File Created. Shape: {sub_df.shape}")
    print(f"  First 3 rows:\n{sub_df.head(3)}")

    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Missing required columns in submission"
    assert len(sub_df) > 0, "Submission file is empty"

    # 6. Demonstrate Standalone Inference Module
    print("\n[Step 5] Running Standalone Inference Module")
    output_dir_inf = "./submission_inference_check"

    # This function is designed to run inference independently of the training session
    # We point it to the model path (even if missing, it handles it gracefully with a warning for demo purposes)
    generate_submission(
        model_path=model_path,
        output_dir=output_dir_inf,
        batch_size=batch_size,
        num_workers=2,
        debug=True,
        img_size=260,
    )

    inf_sub_path = os.path.join(output_dir_inf, "submission.csv")
    assert os.path.exists(inf_sub_path), "Inference module failed to create submission"

    inf_df = pd.read_csv(inf_sub_path)
    print(f"  Inference Submission Created. Shape: {inf_df.shape}")
    assert len(inf_df) > 0, "Inference submission is empty"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
