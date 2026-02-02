import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score
from library.config import (
    PROTOTYPES_PATH,
    CLASS_IDS_PATH,
    IDEA_DIR,
    SUBMISSION_DIR,
    NUM_CLASSES,
)
from library.model import FeatureExtractor
from library.utils import get_device


class PrototypeClassifier:
    """
    A Nearest Class Mean classifier using a fixed feature extractor.
    """

    def __init__(self):
        self.device = get_device()
        self.backbone = FeatureExtractor().to(self.device)
        self.backbone.eval()
        self.prototypes = None
        self.class_ids = None

    def fit(self, train_loader, load_cached_data=True):
        """
        Computes or loads class prototypes from the training data.

        Args:
            train_loader: DataLoader for the training set.
            load_cached_data (bool): If True, attempts to load prototypes from disk.
        """
        # Ensure cache directory exists
        os.makedirs(IDEA_DIR, exist_ok=True)

        if (
            load_cached_data
            and os.path.exists(PROTOTYPES_PATH)
            and os.path.exists(CLASS_IDS_PATH)
        ):
            print("Loading cached prototypes...")
            self.prototypes = torch.from_numpy(np.load(PROTOTYPES_PATH)).to(self.device)
            self.class_ids = torch.from_numpy(np.load(CLASS_IDS_PATH)).to(self.device)
            return

        print("Computing prototypes from training data...")

        # We use a dense tensor to accumulate sums.
        # Max category_id is around 15505, we use a safe buffer or NUM_CLASSES + buffer.
        # The provided config says NUM_CLASSES = 15501, but IDs go up to ~15505.
        # We'll allocate enough space for the max possible ID.
        max_id = 16000
        feature_dim = self.backbone.output_dim

        # Accumulators on device
        class_sums = torch.zeros((max_id, feature_dim), device=self.device)
        class_counts = torch.zeros(max_id, device=self.device)

        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Extract features
                features = self.backbone(images)  # (B, D)

                # Accumulate features by class
                class_sums.index_add_(0, labels, features)
                class_counts.index_add_(
                    0, labels, torch.ones_like(labels, dtype=torch.float)
                )

        # Filter out classes that were not present in the training data
        present_mask = class_counts > 0
        valid_sums = class_sums[present_mask]
        valid_counts = class_counts[present_mask].unsqueeze(1)
        valid_ids = torch.arange(max_id, device=self.device)[present_mask]

        # Compute means
        means = valid_sums / valid_counts

        # L2 Normalize prototypes (for cosine similarity)
        self.prototypes = F.normalize(means, p=2, dim=1)
        self.class_ids = valid_ids

        # Save to cache
        print("Saving prototypes to cache...")
        np.save(PROTOTYPES_PATH, self.prototypes.cpu().numpy())
        np.save(CLASS_IDS_PATH, self.class_ids.cpu().numpy())

    def predict(self, loader, is_test=False):
        """
        Performs inference on a dataloader.

        Args:
            loader: DataLoader for validation or testing.
            is_test (bool): If True, expects (image, image_id) tuples.
                            If False, expects (image, label) tuples.

        Returns:
            tuple: (predictions, auxiliary_data)
                   predictions: List of predicted category_ids.
                   auxiliary_data: List of ground truth labels (if val) or image_ids (if test).
        """
        if self.prototypes is None:
            raise RuntimeError("Model has not been fitted yet.")

        all_preds = []
        all_aux = []

        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                aux = batch[1]  # labels or image_ids

                # Extract and normalize features
                features = self.backbone(images)
                features = F.normalize(features, p=2, dim=1)

                # Cosine similarity: (B, D) @ (N_classes, D).T -> (B, N_classes)
                similarities = torch.mm(features, self.prototypes.t())

                # Get index of best match in the prototype matrix
                max_indices = torch.argmax(similarities, dim=1)

                # Map index back to actual category_id
                pred_category_ids = self.class_ids[max_indices]

                all_preds.extend(pred_category_ids.cpu().tolist())

                # Handle aux data (labels or strings)
                if torch.is_tensor(aux):
                    all_aux.extend(aux.tolist())
                else:
                    all_aux.extend(aux)  # For string image_ids in test

        return all_preds, all_aux

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set and prints the Macro F1 score.
        """
        print("Evaluating on validation set...")
        preds, labels = self.predict(val_loader, is_test=False)

        # Calculate Macro F1
        score = f1_score(labels, preds, average="macro")
        print(f"Validation Macro F1 Score: {score}")
        return score

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        print("Generating submission...")
        preds, image_ids = self.predict(test_loader, is_test=True)

        submission_df = pd.DataFrame({"Id": image_ids, "Predicted": preds})

        # Ensure Id is sorted or formatted correctly if needed, though sample submission
        # usually implies just matching IDs. The sample submission has 'Id' as int.
        # Our dataset returns image_id as string or int depending on metadata.
        # Based on sample_submission.csv, Id is int.
        submission_df["Id"] = submission_df["Id"].astype(int)

        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
