import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.models import get_model
from library.data import get_dataloaders


class TTAEngine:
    """
    Engine for performing Test Time Augmentation (TTA) inference with an ensemble of models.
    """

    def __init__(self, models, device=Config.DEVICE):
        """
        Args:
            models (list): List of loaded PyTorch models.
            device (str): Computation device.
        """
        self.models = models
        self.device = device

        # Ensure models are in evaluation mode
        for model in self.models:
            model.to(self.device)
            model.eval()

    def predict(self, dataloader):
        """
        Generates predictions using TTA and Ensemble averaging.

        TTA Views applied:
        1. Original
        2. Horizontal Flip
        3. Vertical Flip
        4. Rotate 90 degrees

        Args:
            dataloader: DataLoader for the test set.

        Returns:
            np.array: Flattened array of predicted probabilities.
        """
        all_preds = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)

                # Create TTA views directly on GPU tensors
                # Shape: (Batch, C, H, W)
                views = {
                    "orig": images,
                    "hflip": torch.flip(images, dims=[-1]),
                    "vflip": torch.flip(images, dims=[-2]),
                    "rot90": torch.rot90(images, k=1, dims=[-2, -1]),
                }

                batch_ensemble_preds = []

                # Iterate through each model in the ensemble
                for model in self.models:
                    model_view_preds = []

                    # Iterate through each TTA view
                    for _, view_tensor in views.items():
                        logits = model(view_tensor)
                        probs = torch.sigmoid(logits)
                        model_view_preds.append(probs)

                    # Average predictions across the 4 views for this model
                    # Stack -> (4, Batch, 1) -> Mean -> (Batch, 1)
                    avg_model_preds = torch.stack(model_view_preds).mean(dim=0)
                    batch_ensemble_preds.append(avg_model_preds)

                # Average predictions across the ensemble members
                # Stack -> (NumModels, Batch, 1) -> Mean -> (Batch, 1)
                final_batch_preds = torch.stack(batch_ensemble_preds).mean(dim=0)

                # Collect results
                all_preds.extend(final_batch_preds.cpu().numpy().flatten())

        return np.array(all_preds)


def load_ensemble_models(device):
    """
    Loads the trained models specified in the configuration.

    Returns:
        list: A list of instantiated models with loaded weights.
    """
    loaded_models = []

    for model_name in Config.MODEL_NAMES:
        print(f"Loading ensemble member: {model_name}")

        # Instantiate the model architecture
        model = get_model(model_name, pretrained=False, num_classes=Config.NUM_CLASSES)

        # Construct path to the best checkpoint
        weights_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {weights_path}")
        else:
            print(
                f"Warning: Checkpoint not found at {weights_path}. Using random initialization."
            )

        loaded_models.append(model)

    return loaded_models


def run_inference(debug=False, sample_size=1000):
    """
    Orchestrates the inference pipeline: data loading, model loading, TTA prediction, and submission saving.

    Args:
        debug (bool): If True, runs on a subset of the test data.
        sample_size (int): Number of samples to use if debug is True.
    """
    set_seed(Config.SEED)

    print(f"Starting Inference (Debug={debug})...")

    # 1. Prepare Data
    # We use the library function to get the test loader
    dataloaders = get_dataloaders(
        train_path=Config.TRAIN_METADATA_PATH,
        val_path=Config.VAL_METADATA_PATH,
        test_path=Config.TEST_METADATA_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
        sample_size=sample_size,
    )
    test_loader = dataloaders["test"]

    # Load the metadata DataFrame manually to ensure we have the corresponding IDs
    # We must apply the exact same sampling logic as get_dataloaders to ensure alignment
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    if debug:
        test_df = test_df.sample(
            n=min(len(test_df), sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    print(f"Test set size: {len(test_df)} images")

    # 2. Load Ensemble
    models = load_ensemble_models(Config.DEVICE)

    # 3. Run TTA Inference
    print("Running prediction with TTA...")
    engine = TTAEngine(models, Config.DEVICE)
    predictions = engine.predict(test_loader)

    # 4. Generate Submission
    if len(predictions) != len(test_df):
        raise ValueError(
            f"Prediction count mismatch! Expected {len(test_df)}, got {len(predictions)}"
        )

    print("Saving submission...")
    submission_df = pd.DataFrame({"id": test_df["id"], "label": predictions})

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
