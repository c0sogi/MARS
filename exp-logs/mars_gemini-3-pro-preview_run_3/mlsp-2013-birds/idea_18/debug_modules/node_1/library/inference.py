import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.models import ModelFactory
from library.data import get_test_dataloader
from library.utils import average_checkpoints, seed_everything


class InferenceEngine:
    """
    Handles the inference process including model loading, weight averaging,
    Test-Time Augmentation (TTA), and submission generation.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.num_classes = Config.NUM_CLASSES
        self.architectures = Config.ARCHITECTURES
        self.num_folds = Config.NUM_FOLDS

        # ImageNet normalization stats for padding calculation
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        # Value 0 (black) maps to (0 - mean) / std
        self.pad_values = (
            torch.tensor(
                [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], dtype=torch.float32
            )
            .view(1, 3, 1, 1)
            .to(self.device)
        )

    def _get_averaged_weights(self, arch, fold):
        """
        Identifies Top-K checkpoints for a specific arch/fold, averages them,
        and returns the path to the temporary averaged checkpoint.
        """
        # Pattern matches checkpoints saved by Trainer: {arch}_fold_{fold}_epoch_{epoch}.pth
        pattern = os.path.join(Config.CHECKPOINT_DIR, f"{arch}_fold_{fold}_epoch_*.pth")
        checkpoints = glob.glob(pattern)

        if not checkpoints:
            print(f"Warning: No checkpoints found for {arch} fold {fold}. Skipping.")
            return None

        # Output path for averaged weights
        avg_path = os.path.join(Config.CHECKPOINT_DIR, f"avg_{arch}_fold_{fold}.pth")

        # Use provided utility to average weights
        # This function saves the result to avg_path
        average_checkpoints(checkpoints, avg_path)

        return avg_path

    def _apply_tta_and_predict(self, model, images):
        """
        Applies Horizontal Shift TTA: Original, Shift Left, Shift Right.
        Returns averaged probabilities for the batch.
        """
        batch_size, c, h, w = images.shape
        shift_pixels = int(w * 0.1)  # 10% shift

        # 1. Original
        imgs_orig = images

        # 2. Shift Left (content moves left, padding on right)
        imgs_left = torch.ones_like(images) * self.pad_values
        imgs_left[:, :, :, :-shift_pixels] = images[:, :, :, shift_pixels:]

        # 3. Shift Right (content moves right, padding on left)
        imgs_right = torch.ones_like(images) * self.pad_values
        imgs_right[:, :, :, shift_pixels:] = images[:, :, :, :-shift_pixels]

        # Stack for batch processing: [3*B, C, H, W]
        combined_input = torch.cat([imgs_orig, imgs_left, imgs_right], dim=0)

        with torch.no_grad():
            logits = model(combined_input)
            probs = torch.sigmoid(logits)

        # Split back
        p_orig, p_left, p_right = torch.split(probs, batch_size, dim=0)

        # Average predictions
        avg_probs = (p_orig + p_left + p_right) / 3.0

        return avg_probs

    def generate_submission(self):
        """
        Main method to run inference across all models and generate the submission file.
        """
        seed_everything(Config.SEED)

        # 1. Prepare Data
        test_loader = get_test_dataloader()

        # Dictionary to store accumulated probabilities: rec_id -> probability vector
        # Using a dict to handle potential shuffling or ID tracking easily
        # Initialize with the test IDs
        test_ids = []
        for _, rec_ids in test_loader:
            test_ids.extend(rec_ids.numpy())
        test_ids = sorted(list(set(test_ids)))

        # Storage for accumulated predictions (N_samples, N_classes)
        # We map rec_id to index
        id_to_idx = {rid: i for i, rid in enumerate(test_ids)}
        accumulated_preds = np.zeros(
            (len(test_ids), self.num_classes), dtype=np.float32
        )

        model_count = 0

        # 2. Iterate over Ensemble (Architectures x Folds)
        for arch in self.architectures:
            for fold in range(self.num_folds):
                print(f"Processing {arch} - Fold {fold}...")

                # Prepare weights
                weights_path = self._get_averaged_weights(arch, fold)
                if weights_path is None:
                    continue

                # Load Model
                try:
                    model = ModelFactory.create_model(
                        arch, num_classes=self.num_classes, pretrained=False
                    )
                    state_dict = torch.load(weights_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                except Exception as e:
                    print(f"Error loading model {arch} fold {fold}: {e}")
                    continue

                # Inference Loop
                for images, rec_ids in test_loader:
                    images = images.to(self.device)

                    # Predict with TTA
                    batch_probs = self._apply_tta_and_predict(model, images)
                    batch_probs = batch_probs.cpu().numpy()

                    # Accumulate
                    for rid, prob in zip(rec_ids.numpy(), batch_probs):
                        idx = id_to_idx[rid]
                        accumulated_preds[idx] += prob

                model_count += 1

                # Clean up weights file to save space (optional, but good practice)
                if os.path.exists(weights_path):
                    os.remove(weights_path)

                # Free GPU memory
                del model
                torch.cuda.empty_cache()

        if model_count == 0:
            raise RuntimeError("No models were successfully loaded for inference.")

        # 3. Average Predictions
        final_probs = accumulated_preds / model_count

        # 4. Format Submission
        submission_rows = []
        for i, rid in enumerate(test_ids):
            probs = final_probs[i]
            for species_idx in range(self.num_classes):
                # Format: Id = rec_id * 100 + species_number
                row_id = int(rid * 100 + species_idx)
                probability = probs[species_idx]
                submission_rows.append({"Id": row_id, "Probability": probability})

        df_sub = pd.DataFrame(submission_rows)

        # Sort by Id to match sample submission structure usually expected
        df_sub = df_sub.sort_values("Id").reset_index(drop=True)

        # Save
        out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")
        print(f"Ensemble size: {model_count} models")
