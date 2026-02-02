import os
import pandas as pd
import numpy as np
import torch
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import seed_everything, rank_normalize
from library.dataset import get_test_loader
from library.model import DiseaseClassifier


class InferencePipeline:
    """
    Pipeline for generating predictions using trained SWA models.
    Implements Rank-Calibrated Inference with Test Time Augmentation (TTA).
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

    def reconstruct_probs(self, r, s):
        """
        Reconstructs 4-class probabilities from Rust (r) and Scab (s) scores (ranks).
        Mapping:
            Healthy: (1-r)(1-s)
            Multiple: r*s
            Rust: r(1-s)
            Scab: (1-r)s

        Args:
            r (np.ndarray): Normalized rank scores for Rust.
            s (np.ndarray): Normalized rank scores for Scab.

        Returns:
            np.ndarray: Array of shape (N, 4) with columns [Healthy, Multiple, Rust, Scab].
        """
        healthy = (1 - r) * (1 - s)
        multiple = r * s
        rust_only = r * (1 - s)
        scab_only = (1 - r) * s

        # Stack columns: Healthy, Multiple, Rust, Scab
        return np.stack([healthy, multiple, rust_only, scab_only], axis=1)

    def run(self, test_df: pd.DataFrame):
        """
        Executes the inference pipeline.

        Args:
            test_df (pd.DataFrame): DataFrame containing test image metadata.
        """
        print("Starting Inference Pipeline...")

        all_model_preds = (
            []
        )  # List to store rank-normalized predictions from each model

        # Iterate over defined model architectures
        for model_conf in Config.MODEL_CONFIGS:
            model_name = model_conf["name"]
            img_size = model_conf["img_size"]
            batch_size = model_conf["batch_size"]

            print(f"Processing architecture: {model_name}")

            # Create DataLoader for this specific image size
            test_loader = get_test_loader(
                test_df, img_size=img_size, batch_size=batch_size
            )

            # Iterate over folds
            for fold in range(Config.N_FOLDS):
                # Construct path to SWA model
                checkpoint_path = os.path.join(
                    Config.WORKING_DIR, f"swa_model_{model_name}_fold_{fold}.pth"
                )

                if not os.path.exists(checkpoint_path):
                    print(
                        f"Warning: Checkpoint not found at {checkpoint_path}. Skipping."
                    )
                    continue

                # Initialize model
                model = DiseaseClassifier(model_name=model_name, pretrained=False)
                model.load_weights(checkpoint_path, device=self.device)
                model.to(self.device)
                model.eval()

                fold_raw_preds = []

                # Inference loop
                with torch.no_grad():
                    for imgs, _ in test_loader:
                        imgs = imgs.to(self.device)

                        # 1. Forward pass original
                        with autocast():
                            logits = model(imgs)
                            probs = torch.sigmoid(logits)

                        # 2. Forward pass TTA (Horizontal Flip)
                        imgs_flip = torch.flip(imgs, dims=[3])
                        with autocast():
                            logits_flip = model(imgs_flip)
                            probs_flip = torch.sigmoid(logits_flip)

                        # Average TTA
                        avg_probs = (probs + probs_flip) / 2.0
                        fold_raw_preds.append(avg_probs.cpu().numpy())

                # Concatenate batches for this fold
                if fold_raw_preds:
                    fold_raw_preds = np.concatenate(
                        fold_raw_preds, axis=0
                    )  # Shape (N, 2)

                    # Apply Rank Normalization
                    # We normalize the predictions of this specific model/fold relative to the test set
                    # This handles calibration differences between architectures/folds
                    ranked_preds = rank_normalize(fold_raw_preds)
                    all_model_preds.append(ranked_preds)

                print(f"  Fold {fold} processed.")

        if not all_model_preds:
            raise RuntimeError(
                "No predictions generated. Ensure models are trained and checkpoints exist."
            )

        # Average the ranks across all models (Ensemble)
        # all_model_preds is a list of (N, 2) arrays
        avg_ranks = np.mean(all_model_preds, axis=0)  # Shape (N, 2)

        # Extract Rust and Scab ranks
        r_ranks = avg_ranks[:, 0]
        s_ranks = avg_ranks[:, 1]

        # Reconstruct 4-class probabilities
        # Returns shape (N, 4) -> [Healthy, Multiple, Rust, Scab]
        final_probs = self.reconstruct_probs(r_ranks, s_ranks)

        # Create Submission DataFrame
        # Columns must match sample_submission.csv format
        submission_df = pd.DataFrame(
            {
                "image_id": test_df["image_id"],
                "healthy": final_probs[:, 0],
                "multiple_diseases": final_probs[:, 1],
                "rust": final_probs[:, 2],
                "scab": final_probs[:, 3],
            }
        )

        # Save submission
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        # Also save to root as per standard convention
        submission_df.to_csv("submission.csv", index=False)

        return submission_df
