import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from library.utils import seed_everything, get_device
from library.dataset import load_and_cache_images
from library.models import BirdModel

# Constants
CACHE_DIR = "./working/idea_16/"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
NUM_CLASSES = 19
FOLDS = 5
ARCHITECTURES = ["resnet18", "densenet121"]
TTA_SHIFTS = [0.0, 0.25, 0.5, 0.75]  # 0%, 25%, 50%, 75% time roll


class TTADataset(Dataset):
    """
    Dataset class specifically for Test-Time Augmentation.
    Applies deterministic time-rolling based on shift_pct.
    """

    def __init__(self, df, image_dict, height, width, shift_pct=0.0):
        self.df = df.reset_index(drop=True)
        self.image_dict = image_dict
        self.shift_pct = shift_pct

        # Define transforms (Resize -> Normalize -> ToTensor)
        # Note: Gray2RGB and Roll happen before this pipeline
        self.transforms = A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])

        # Retrieve image
        if rec_id in self.image_dict:
            img = self.image_dict[rec_id].copy()
        else:
            # Fallback (should not happen)
            img = np.zeros((256, 1246), dtype=np.uint8)

        # 1. Deterministic Time Roll (TTA)
        if self.shift_pct > 0.0:
            shift = int(img.shape[1] * self.shift_pct)
            img = np.roll(img, shift, axis=1)

        # 2. Pseudo-RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 3. Albumentations
        augmented = self.transforms(image=img)
        img = augmented["image"]

        return {"id": rec_id, "image": img}


def predict_with_model(model, loader, device):
    """
    Runs inference for a single model/TTA configuration.
    Returns a dictionary mapping rec_id to probability vectors.
    """
    model.eval()
    preds_dict = {}

    with torch.no_grad():
        for batch in loader:
            ids = batch["id"].numpy()
            images = batch["image"].to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, rec_id in enumerate(ids):
                preds_dict[rec_id] = probs[i]

    return preds_dict


def run_inference(
    models_dir="./working/idea_16",
    output_dir="./submission",
    batch_size=32,
    num_workers=2,
    load_cached_data=True,
):
    """
    Main inference function.
    Aggregates predictions from 3 architectures * 5 folds * 4 TTA shifts.
    """
    seed_everything(42)
    device = get_device()

    # 1. Load Metadata and Images
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {TEST_CSV}")

    df_test = pd.read_csv(TEST_CSV)
    print(f"Loaded test metadata: {len(df_test)} samples")

    # Use library function to load images (handles caching)
    image_dict = load_and_cache_images(df_test, CACHE_DIR, load_cached_data)

    # Initialize accumulator for ensemble predictions
    # Map: rec_id -> np.array(shape=(19,))
    ensemble_preds = {
        rec_id: np.zeros(NUM_CLASSES) for rec_id in df_test["rec_id"].unique()
    }

    total_models_count = 0

    # 2. Iterate Architectures
    for arch in ARCHITECTURES:
        # Determine resolution
        if "densenet" in arch:
            height, width = 160, 320
        else:
            height, width = 224, 448

        print(f"\nProcessing Architecture: {arch} (Resolution: {height}x{width})")

        # 3. Iterate Folds
        for fold in range(FOLDS):
            model_path = os.path.join(models_dir, f"model_{arch}_fold_{fold}.pth")

            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found. Skipping.")
                continue

            # Load Model
            try:
                model = BirdModel(
                    model_name=arch, num_classes=NUM_CLASSES, pretrained=False
                )
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue

            # 4. Iterate TTA Shifts
            for shift_pct in TTA_SHIFTS:
                # Create TTA Dataset and Loader
                dataset = TTADataset(
                    df_test, image_dict, height, width, shift_pct=shift_pct
                )
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )

                # Predict
                preds = predict_with_model(model, loader, device)

                # Accumulate
                for rec_id, prob_vec in preds.items():
                    ensemble_preds[rec_id] += prob_vec

                total_models_count += 1

            # Clean up memory
            del model
            torch.cuda.empty_cache()

    if total_models_count == 0:
        raise RuntimeError("No models were successfully loaded and executed.")

    print(
        f"\nInference complete. Aggregated {total_models_count} predictions (Models x TTA)."
    )

    # 5. Average and Format Submission
    submission_rows = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")

    # Sort rec_ids to ensure deterministic order (though not strictly required by format)
    sorted_rec_ids = sorted(ensemble_preds.keys())

    for rec_id in sorted_rec_ids:
        # Average probabilities
        avg_probs = ensemble_preds[rec_id] / total_models_count

        # Create rows for each species
        for species_idx in range(NUM_CLASSES):
            # Id format: rec_id * 100 + species_id
            row_id = rec_id * 100 + species_idx
            prob = avg_probs[species_idx]
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_submission = pd.DataFrame(submission_rows)

    # Save
    df_submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(df_submission.head())
