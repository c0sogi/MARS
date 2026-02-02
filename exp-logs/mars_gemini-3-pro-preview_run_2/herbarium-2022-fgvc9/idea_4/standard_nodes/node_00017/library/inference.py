import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

from library.config import Config
from library.utils import get_logger
from library.model import CascadedTaxonomicNetwork
from library.dataset import get_dataloaders

logger = get_logger(__name__)


class InferenceRunner:
    """
    Handles the inference lifecycle for the Cascaded Taxonomic Network.
    Performs predictions on the test set, applies Test Time Augmentation (TTA),
    and generates the submission file.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.submission_path = Config.SUBMISSION_PATH
        self.weights_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    def _load_resources(self):
        """
        Loads the test dataloader and metadata counts required for model initialization.
        """
        logger.info("Loading DataLoaders and Taxonomy...")
        # Use Config.DEBUG to determine if we should load the full set or a subset
        # Note: get_dataloaders computes training weights internally, which we accept
        # as we cannot modify the library files.
        _, _, test_loader, meta_counts = get_dataloaders(debug=Config.DEBUG)
        return test_loader, meta_counts

    def _initialize_model(self, meta_counts):
        """
        Initializes the model architecture and loads the trained weights.
        """
        num_species = Config.NUM_CLASSES
        num_families = meta_counts["num_families"]
        num_genera = meta_counts["num_genera"]

        logger.info(
            f"Initializing model: {num_species} Species, {num_genera} Genera, {num_families} Families."
        )

        model = CascadedTaxonomicNetwork(
            num_species=num_species,
            num_families=num_families,
            num_genera=num_genera,
            pretrained=False,  # No need to download pretrained weights, we load our own
        ).to(self.device)

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights not found at {self.weights_path}")

        logger.info(f"Loading state dict from {self.weights_path}")
        state_dict = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def _save_submission(self, ids, preds):
        """
        Formats and saves the predictions to a CSV file.
        """
        df = pd.DataFrame({"Id": ids, "Predicted": preds})

        # Sort by Id to ensure consistency with submission requirements
        # Attempt to convert to int for numerical sorting, fallback to string if fails
        try:
            df["Id_int"] = df["Id"].astype(int)
            df = df.sort_values("Id_int")
            df = df.drop(columns=["Id_int"])
        except ValueError:
            df = df.sort_values("Id")

        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        df.to_csv(self.submission_path, index=False)
        logger.info(f"Submission saved to {self.submission_path}")

        # Log first few rows for verification
        logger.info(f"Submission Head:\n{df.head().to_string()}")

    def run(self):
        """
        Main execution method for inference.
        """
        try:
            # 1. Prepare Data and Model
            test_loader, meta_counts = self._load_resources()
            model = self._initialize_model(meta_counts)

            predictions = []
            image_ids = []

            logger.info("Starting Inference with TTA (Original + Horizontal Flip)...")

            # 2. Inference Loop
            with torch.no_grad():
                for i, (images, ids) in enumerate(test_loader):
                    images = images.to(self.device)

                    # --- TTA View 1: Original ---
                    # labels=None triggers ArcFace to return scaled cosine similarity logits
                    logits_original, _, _ = model(images, labels=None)

                    # --- TTA View 2: Horizontal Flip ---
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped, _, _ = model(images_flipped, labels=None)

                    # --- Aggregate ---
                    # Average the cosine similarities
                    avg_logits = (logits_original + logits_flipped) / 2.0

                    # Get predicted class index
                    _, preds = torch.max(avg_logits, 1)

                    predictions.extend(preds.cpu().numpy())
                    image_ids.extend(ids)

                    # Periodic logging (avoid progress bar)
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed batch {i + 1}/{len(test_loader)}")

            # 3. Save Results
            self._save_submission(image_ids, predictions)
            logger.info("Inference completed successfully.")

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise e
