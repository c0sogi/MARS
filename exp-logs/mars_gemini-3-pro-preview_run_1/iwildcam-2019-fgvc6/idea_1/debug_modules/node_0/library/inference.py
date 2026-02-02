import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import AnimalDataset, get_transforms
from library.model import get_model


def generate_submission(
    weights_path=Config.MODEL_SAVE_PATH, batch_size=Config.BATCH_SIZE, debug=False
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        weights_path (str): Path to the trained model weights.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    # Ensure reproducibility
    Config.set_seed(Config.SEED)

    # Setup device
    device = torch.device(Config.DEVICE)

    # Load Test Metadata
    if not os.path.exists(Config.TEST_META_PATH):
        raise FileNotFoundError(
            f"Test metadata file not found at {Config.TEST_META_PATH}"
        )

    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Handle Debug Mode
    if debug:
        print("Running in DEBUG mode: Sampling 100 test images.")
        df_test = df_test.sample(
            n=min(len(df_test), 100), random_state=Config.SEED
        ).reset_index(drop=True)

    # Prepare Dataset and Loader
    # Use 'val' transforms for testing (resize + normalize, no augmentations)
    test_transform = get_transforms(mode="val")

    # Initialize Dataset
    # We manually create the dataset to ensure strict alignment between df_test rows and loader iterations
    test_dataset = AnimalDataset(
        df=df_test, root_dir=Config.INPUT_DIR, transform=test_transform
    )

    # Initialize DataLoader
    # shuffle=False is critical to maintain order
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    print(f"Loading model weights from {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. Please train the model first."
        )

    model = get_model(device=device, weights_path=weights_path)
    model.eval()

    all_preds = []

    # Inference Loop
    print(f"Starting inference on {len(df_test)} images...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted class indices (argmax of logits)
            preds = torch.argmax(outputs, dim=1)

            # Collect predictions
            all_preds.extend(preds.cpu().numpy())

    # Generate Submission DataFrame
    # The 'Id' column comes directly from the dataframe used to create the loader
    submission_df = pd.DataFrame({"Id": df_test["Id"], "Predicted": all_preds})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved successfully to {submission_path}")
    print("First 5 predictions:")
    print(submission_df.head())
