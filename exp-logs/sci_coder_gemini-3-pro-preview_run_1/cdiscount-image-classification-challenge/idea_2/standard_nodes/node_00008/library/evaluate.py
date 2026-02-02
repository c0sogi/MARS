import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import BSONProductDataset, collate_fn
from library.model import HierarchicalResNet50
from library.utils import CategoryHierarchy


class Evaluator:
    """
    Handles validation and inference for the Hierarchical ResNet-50.
    """

    def __init__(self, model, device):
        """
        Args:
            model (nn.Module): The loaded model.
            device (torch.device): Device to run inference on.
        """
        self.model = model
        self.device = device
        self.hierarchy = CategoryHierarchy(load_cached_data=True)

    def validate(self, val_loader):
        """
        Computes accuracy on the validation set using the Level 3 (Target) head.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            float: Validation accuracy (0.0 to 1.0).
        """
        self.model.eval()
        correct_l3 = 0
        total_samples = 0

        print("Starting Validation...")
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch["images"].to(self.device)
                targets_l3 = batch["labels"]["target"].to(self.device)

                # Forward Pass
                outputs = self.model(images)

                # Get predictions from the target head (Level 3)
                _, preds = torch.max(outputs["target"], 1)

                # Update metrics
                correct_l3 += torch.sum(preds == targets_l3).item()
                total_samples += images.size(0)

        accuracy = correct_l3 / total_samples if total_samples > 0 else 0.0
        print(f"Validation Accuracy (Level 3): {accuracy}")
        return accuracy

    def generate_submission(self, test_loader, output_path):
        """
        Generates predictions for the test set and saves them to a CSV file.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            output_path (str): Path to save the submission CSV.
        """
        self.model.eval()
        results = []

        print("Starting Inference on Test Set...")
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                images = batch["images"].to(self.device)
                ids = batch["ids"].numpy()  # CPU numpy array for storage

                # Forward Pass
                outputs = self.model(images)

                # Get predictions from the target head (Level 3)
                # We only care about the fine-grained category for submission
                _, preds_indices = torch.max(outputs["target"], 1)
                preds_indices = preds_indices.cpu().numpy()

                # Map internal indices back to original category_ids
                for pid, pred_idx in zip(ids, preds_indices):
                    category_id = self.hierarchy.get_category_id_from_l3(pred_idx)
                    results.append({"_id": pid, "category_id": category_id})

        # Create DataFrame
        df_submission = pd.DataFrame(results)

        # Ensure correct column order
        df_submission = df_submission[["_id", "category_id"]]

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}. Total records: {len(df_submission)}")


def load_model_weights(model, checkpoint_path, device):
    """
    Loads model weights from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )
        return model

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def evaluate_model(run_validation=True, run_submission=True):
    """
    Main entry point to run evaluation and submission generation.

    Args:
        run_validation (bool): Whether to run validation on the val set.
        run_submission (bool): Whether to generate predictions on the test set.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model
    model = HierarchicalResNet50()
    model.to(device)

    # Load best weights
    model = load_model_weights(model, Config.MODEL_CHECKPOINT, device)

    # 3. Initialize Evaluator
    evaluator = Evaluator(model, device)

    # 4. Run Validation
    if run_validation:
        print("Initializing Validation Loader...")
        val_dataset = BSONProductDataset(mode="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        evaluator.validate(val_loader)

    # 5. Run Submission
    if run_submission:
        print("Initializing Test Loader...")
        test_dataset = BSONProductDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        evaluator.generate_submission(test_loader, Config.SUBMISSION_PATH)
