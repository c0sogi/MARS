import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import RetinopathyDataset, get_transforms
from library.models import RetinopathyModel
from library.utils import seed_everything


def predict_and_submit(debug=False):
    """
    Generates predictions for the test set using the trained ensemble and creates a submission file.

    Implements the Multi-Scale Heterogeneous Ensemble inference strategy:
    1. Iterates through defined architectures (EfficientNet-B5, ConvNeXt).
    2. Loads models for all 5 folds.
    3. Applies Test-Time Augmentation (Original + Horizontal Flip).
    4. Aggregates continuous regression scores via averaging.
    5. Rounds and formats for final submission.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data for verification.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("Starting Inference...")

    # Load test metadata
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

    df_test = pd.read_csv(Config.TEST_CSV)

    # Handle Debug Mode
    if debug:
        print("Debug mode enabled: Processing first 10 samples only.")
        df_test = df_test.head(10)

    # We preserve the order of IDs for the submission dataframe
    test_ids = df_test["id_code"].values
    num_samples = len(df_test)

    # Accumulator for predictions (Soft Voting)
    # We sum up the continuous scores from all models
    accumulated_scores = torch.zeros(num_samples, dtype=torch.float32)
    models_executed = 0

    # Iterate through defined architectures and their input sizes
    # Config.MODEL_SPECS = {"tf_efficientnet_b5_ns": 512, "convnext_base": 384}
    for model_name, image_size in Config.MODEL_SPECS.items():
        print(f"\nProcessing Architecture: {model_name} (Input Size: {image_size})")

        # Create a temporary CSV for the dataset class if needed (e.g., debug slicing)
        current_csv_path = Config.TEST_CSV
        if debug:
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            current_csv_path = os.path.join(Config.OUTPUT_DIR, "test_debug.csv")
            df_test.to_csv(current_csv_path, index=False)

        # Initialize Dataset and DataLoader for the specific image size
        test_dataset = RetinopathyDataset(
            csv_path=current_csv_path,
            transform=get_transforms(image_size, mode="test"),
            mode="test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,  # Must be False to preserve order relative to test_ids
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Iterate through all folds for this architecture
        for fold in range(Config.NUM_FOLDS):
            checkpoint_name = f"{model_name}_fold_{fold}.pth"
            checkpoint_path = os.path.join(Config.OUTPUT_DIR, checkpoint_name)

            # Skip if checkpoint is missing (allows for partial ensemble evaluation)
            if not os.path.exists(checkpoint_path):
                print(f"  > Checkpoint not found: {checkpoint_name}. Skipping.")
                continue

            print(f"  > Loading Fold {fold} model...")

            # Initialize model structure
            model = RetinopathyModel(model_name=model_name, pretrained=False)

            # Load trained weights
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            fold_preds = []

            # Inference Loop
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)

                    # 1. Forward Pass (Original)
                    out_orig = model(images)

                    # 2. Forward Pass (Test-Time Augmentation: Horizontal Flip)
                    # Flip along width axis (dim 3 for NCHW tensor)
                    images_flipped = torch.flip(images, dims=[3])
                    out_flip = model(images_flipped)

                    # 3. Average TTA predictions
                    batch_preds = (out_orig + out_flip) / 2.0

                    # Store result on CPU to save GPU memory
                    fold_preds.append(batch_preds.cpu())

            # Concatenate results for this fold
            fold_preds_tensor = torch.cat(fold_preds, dim=0)

            # Accumulate into global scores
            accumulated_scores += fold_preds_tensor
            models_executed += 1

            # Cleanup to free GPU memory for the next model
            del model
            torch.cuda.empty_cache()

    if models_executed == 0:
        print("Error: No models were executed. Submission generation failed.")
        return

    # Average scores across all executed models
    final_scores = accumulated_scores / models_executed

    # Post-processing
    # 1. Round continuous regression score to nearest integer
    # 2. Clamp to valid class range [0, 4]
    # 3. Convert to integer type
    final_preds = torch.round(final_scores).int()
    final_preds = torch.clamp(final_preds, 0, 4).numpy()

    # Create Submission DataFrame
    submission = pd.DataFrame({"id_code": test_ids, "diagnosis": final_preds})

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"\nInference completed.")
    print(f"Ensembled {models_executed} models.")
    print(f"Submission saved to: {submission_path}")

    # Cleanup temporary debug file
    if debug and os.path.exists(os.path.join(Config.OUTPUT_DIR, "test_debug.csv")):
        os.remove(os.path.join(Config.OUTPUT_DIR, "test_debug.csv"))
