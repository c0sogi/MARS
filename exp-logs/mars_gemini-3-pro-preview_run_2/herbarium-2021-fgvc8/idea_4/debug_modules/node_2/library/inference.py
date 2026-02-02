import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
from library.config import Config
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt


class InferenceEngine:
    """
    Engine for running inference on the test dataset and generating submission files.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = None

    def _load_model(self, num_families, num_orders, checkpoint_path):
        """
        Initializes the model architecture and loads weights from the checkpoint.

        Args:
            num_families (int): Number of family classes for the auxiliary head.
            num_orders (int): Number of order classes for the auxiliary head.
            checkpoint_path (str): Path to the model state dictionary.
        """
        print(
            f"Initializing model with {num_families} families and {num_orders} orders..."
        )
        # We set pretrained=False because we are loading our own fine-tuned weights
        self.model = HierarchicalConvNeXt(
            num_families=num_families, num_orders=num_orders, pretrained=False
        )
        self.model.to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def generate_submission(
        self,
        checkpoint_path=Config.MODEL_CHECKPOINT,
        output_path=Config.SUBMISSION_FILE,
    ):
        """
        Runs inference on the test set and saves the submission CSV.

        Args:
            checkpoint_path (str): Path to the trained model weights.
            output_path (str): Path where the submission CSV will be saved.
        """
        # 1. Get DataLoaders and Taxonomy Info
        # We reuse get_dataloaders to ensure consistent preprocessing and transforms.
        # Stage 1 is sufficient as it returns the test loader and taxonomy counts.
        print("Preparing data loaders...")
        loaders, num_families, num_orders = get_dataloaders(stage=1)
        test_loader = loaders["test"]

        # 2. Load Model
        self._load_model(num_families, num_orders, checkpoint_path)

        # 3. Inference Loop
        image_ids_list = []
        predictions_list = []

        print(f"Starting inference on {len(test_loader.dataset)} test images...")

        with torch.no_grad():
            for batch_idx, (images, image_ids) in enumerate(test_loader):
                images = images.to(self.device, non_blocking=True)

                # Use autocast for mixed precision inference (faster on A100)
                with autocast():
                    species_logits, _, _ = self.model(images)

                # Get predicted class (argmax of species logits)
                preds = torch.argmax(species_logits, dim=1).cpu().numpy()

                image_ids_list.extend(image_ids.numpy())
                predictions_list.extend(preds)

        # 4. Create Submission DataFrame
        df_submission = pd.DataFrame(
            {"Id": image_ids_list, "Predicted": predictions_list}
        )

        # Ensure sorted by Id as per submission format requirements
        df_submission.sort_values(by="Id", inplace=True)

        # 5. Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_submission.to_csv(output_path, index=False)

        print(f"Inference complete. Submission saved to {output_path}")
        print(f"Total predictions generated: {len(df_submission)}")
