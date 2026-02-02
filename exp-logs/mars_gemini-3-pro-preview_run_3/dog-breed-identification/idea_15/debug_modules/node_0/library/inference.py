import os
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.dataset import get_test_dataloader
from library.model import DogModel
from library.utils import seed_everything


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    This function predicts probabilities for the original image and a horizontally
    flipped version, then averages them to improve robustness.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): Test dataloader.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Averaged probabilities (N_samples, N_classes).
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            # 1. Original Image Prediction
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # 2. Flipped Image Prediction (TTA)
            # Flip width dimension (dim 3 for NCHW tensor)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.softmax(logits_flipped, dim=1)

            # 3. Average Probabilities (Level 2 Ensemble)
            avg_probs = (probs + probs_flipped) / 2.0
            all_probs.append(avg_probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def predict_and_submit():
    """
    Executes the hierarchical inference pipeline.

    Steps:
    1. Loads test data and class metadata.
    2. Iterates through all 5 folds.
    3. Loads the 'Manual Soup' model for each fold (Level 1 Ensemble).
    4. Performs TTA inference for each model (Level 2 Ensemble).
    5. Averages predictions across all folds (Level 3 Ensemble / Bagging).
    6. Formats and saves the submission CSV.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Inference on device: {device}")

    # 1. Load Test Data
    # get_test_dataloader returns the loader (shuffle=False) and the dataframe (for IDs)
    test_loader, test_df = get_test_dataloader()

    # 2. Load Class Mapping
    # We need to map the model's output indices (0..119) back to breed names
    classes_path = os.path.join(Config.OUTPUT_DIR, "classes.parquet")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(
            f"classes.parquet not found at {classes_path}. "
            "Ensure training has been run to generate metadata."
        )

    classes_df = pd.read_parquet(classes_path)
    # Create mapping: idx -> breed name
    idx_to_class = {row["idx"]: row["breed"] for _, row in classes_df.iterrows()}

    # 3. Initialize Ensemble Accumulator
    num_test = len(test_df)
    num_classes = Config.NUM_CLASSES
    ensemble_probs = np.zeros((num_test, num_classes))
    valid_folds = 0

    # 4. Iterate over Folds and Aggregate Predictions
    for fold in range(Config.N_FOLDS):
        soup_path = os.path.join(Config.OUTPUT_DIR, f"best_soup_fold_{fold}.pth")

        if not os.path.exists(soup_path):
            print(
                f"Warning: Soup model for fold {fold} not found at {soup_path}. Skipping."
            )
            continue

        print(f"Predicting with Fold {fold} Soup Model...")

        # Initialize Model
        # pretrained=False because we are loading a full state_dict from disk
        model = DogModel(pretrained=False)

        # Load weights
        state_dict = torch.load(soup_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Predict with TTA
        probs = predict_with_tta(model, test_loader, device)

        # Accumulate
        ensemble_probs += probs
        valid_folds += 1

        # Cleanup to save memory
        del model, state_dict
        torch.cuda.empty_cache()

    if valid_folds == 0:
        raise RuntimeError(
            "No valid models found for inference. Cannot generate submission."
        )

    # 5. Average Predictions (Bagging)
    ensemble_probs /= valid_folds

    # 6. Create Submission File
    print("Creating submission file...")
    submission = pd.DataFrame()
    submission["id"] = test_df["id"]

    # Map probabilities to correct breed columns
    # Ensure we iterate in order of indices 0..N to match the probability array columns
    for idx in range(num_classes):
        if idx not in idx_to_class:
            raise KeyError(f"Index {idx} not found in class mapping.")
        breed_name = idx_to_class[idx]
        submission[breed_name] = ensemble_probs[:, idx]

    # 7. Save to Disk
    os.makedirs("submission", exist_ok=True)
    sub_path = os.path.join("submission", "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
