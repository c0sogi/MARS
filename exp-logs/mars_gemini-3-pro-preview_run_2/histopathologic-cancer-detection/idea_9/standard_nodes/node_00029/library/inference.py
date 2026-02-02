import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.models import PathologyClassifier
from library.dataset import get_dataloaders
from library.training import set_seed


class InferenceEngine:
    """
    Manages the inference process for the heterogeneous ensemble.
    Handles model loading, TTA, and prediction aggregation.
    """

    def __init__(self, checkpoint_dir, device):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.models = self._load_ensemble()

    def _load_ensemble(self):
        """
        Loads all valid model checkpoints from the checkpoint directory.
        Identifies architecture based on filename and initializes the model.
        """
        models = []
        if not os.path.exists(self.checkpoint_dir):
            print(
                f"Warning: Checkpoint directory {self.checkpoint_dir} does not exist."
            )
            return models

        # List all .pth files
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pth")]

        # Filter for 'best_model' checkpoints to ensure we use converged weights
        # (or 'best_convnext' etc. depending on saving logic, assuming 'best_' prefix)
        checkpoint_files = [f for f in files if "best_" in f]

        if not checkpoint_files:
            print("No checkpoints found. Checking for any .pth files...")
            checkpoint_files = files

        print(f"Found {len(checkpoint_files)} checkpoints for ensemble.")

        for filename in checkpoint_files:
            # Determine architecture
            arch = None
            for backbone in Config.MODEL_BACKBONES:
                if backbone in filename:
                    arch = backbone
                    break

            if arch is None:
                print(f"Skipping {filename}: Could not identify architecture.")
                continue

            # Initialize model
            try:
                model = PathologyClassifier(
                    model_name=arch, num_classes=Config.NUM_CLASSES, pretrained=False
                )
                path = os.path.join(self.checkpoint_dir, filename)

                # Load weights
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)

                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"Loaded {arch} from {filename}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

        return models

    def _get_tta_batch(self, images):
        """
        Generates a batch of 8 Dihedral transformations for each image.
        Views: 4 Rotations x 2 Flips.
        Input: (B, C, H, W)
        Output: (8*B, C, H, W)
        """
        augments = []

        # Standard Dihedral Group D4
        # 1. Base images (Identity)
        augments.append(images)
        # 2. Rotations 90, 180, 270
        augments.append(torch.rot90(images, 1, [2, 3]))
        augments.append(torch.rot90(images, 2, [2, 3]))
        augments.append(torch.rot90(images, 3, [2, 3]))

        # 3. Horizontal Flip
        img_flip = torch.flip(images, [3])
        augments.append(img_flip)
        # 4. Rotations of the flipped image
        augments.append(torch.rot90(img_flip, 1, [2, 3]))
        augments.append(torch.rot90(img_flip, 2, [2, 3]))
        augments.append(torch.rot90(img_flip, 3, [2, 3]))

        # Stack along batch dimension
        return torch.cat(augments, dim=0)

    def predict_batch(self, images):
        """
        Runs inference on a batch of images using the ensemble and TTA.
        Returns average probabilities for the batch.
        """
        if not self.models:
            # Fallback for debugging if no models are loaded
            return torch.zeros(images.size(0), 1, device=self.device)

        # Apply TTA: Create augmented batch
        # Shape becomes (8*B, C, H, W)
        tta_batch = self._get_tta_batch(images).to(self.device)

        batch_size = images.size(0)
        num_views = 8
        total_probs = torch.zeros(batch_size, 1, device=self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for model in self.models:
                    # Forward pass
                    logits = model(tta_batch)
                    probs = torch.sigmoid(logits)

                    # Reshape to separate views: (8*B, 1) -> (8, B, 1)
                    probs_views = probs.view(num_views, batch_size, 1)

                    # Average over TTA views
                    model_avg = probs_views.mean(dim=0)

                    # Accumulate model contribution
                    total_probs += model_avg

        # Average over ensemble models
        ensemble_avg = total_probs / len(self.models)
        return ensemble_avg

    def run_inference(self, loader):
        """
        Iterates over the data loader and generates predictions.
        """
        all_ids = []
        all_probs = []

        print(f"Starting inference on {len(loader.dataset)} images...")

        for images, _, ids in loader:
            images = images.to(self.device)

            # Predict
            probs = self.predict_batch(images)

            # Store results
            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

        # Concatenate results
        all_probs = np.concatenate(all_probs, axis=0).flatten()
        return all_ids, all_probs


def generate_submission(load_cached_data=True, debug_size=None):
    """
    Main entry point for generating the submission file.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # We only need the test loader
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    # 3. Model Loading
    engine = InferenceEngine(Config.CHECKPOINT_DIR, device)

    if not engine.models:
        print("Error: No models loaded. Cannot generate submission.")
        # In a real scenario, we might want to fail hard or output dummy predictions.
        # For this task, we proceed to generate a placeholder if needed, but ideally we stop.
        if debug_size is None:
            raise RuntimeError(
                "Ensemble is empty. Checkpoint directory might be missing valid models."
            )

    # 4. Inference
    ids, probs = engine.run_inference(test_loader)

    # 5. Create Submission DataFrame
    df_submission = pd.DataFrame({"id": ids, "label": probs})

    # 6. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Use high precision for float formatting
    df_submission.to_csv(submission_path, index=False, float_format="%.15f")

    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(df_submission)}")
    print(df_submission.head())
