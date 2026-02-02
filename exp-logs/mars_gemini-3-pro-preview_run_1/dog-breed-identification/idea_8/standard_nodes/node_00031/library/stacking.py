import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from library.config import Config
from library.utils import seed_everything, Logger, calculate_metric
from library.dataset import get_dataloaders, get_test_dataloader, get_label_map
from library.models import create_model
from library.inference import predict_tta


class Stacker:
    """
    Implements simple averaging ensemble logic.
    1. Generates/Loads OOF predictions and Test predictions from Expert Model.
    2. Averages predictions across folds.
    """

    def __init__(self, debug=False):
        """
        Args:
            debug (bool): If True, runs in debug mode (fewer folds/data).
        """
        self.debug = debug
        self.work_dir = Config.WORK_DIR
        self.device = torch.device(Config.DEVICE)
        self.logger = Logger(os.path.join(self.work_dir, "stacking.log"))

        # Templates for model weights
        self.model_a_path_fmt = os.path.join(self.work_dir, "convnext_base_fold_{}.pth")

        # Cache file for OOF and Test predictions
        self.cache_path = os.path.join(self.work_dir, "stacking_data.npy")

    def _load_model(self, model_name, checkpoint_path):
        """
        Helper to instantiate a model and load its weights.
        """
        model = create_model(
            model_name, num_classes=Config.NUM_CLASSES, pretrained=False
        )

        if not os.path.exists(checkpoint_path):
            self.logger.log(f"Error: Checkpoint not found at {checkpoint_path}")
            return None

        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # Strip 'module.' prefix if present (caused by SWA AveragedModel wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            # Cite debug_lesson_7: Filter out SWA-specific metadata keys
            if k == "n_averaged":
                continue

            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def get_data(self, load_cached_data=True):
        """
        Retrieves OOF and Test predictions.
        Uses caching to avoid re-running inference on experts.

        Returns:
            dict: Contains 'oof_preds', 'oof_targets', 'oof_ids',
                  'test_preds', 'test_ids'.
        """
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path) and not self.debug:
            self.logger.log(f"Loading cached stacking data from {self.cache_path}")
            try:
                data = np.load(self.cache_path, allow_pickle=True).item()
                return data
            except Exception as e:
                self.logger.log(f"Failed to load cache: {e}. Regenerating...")

        self.logger.log("Generating OOF and Test predictions from scratch...")

        # 2. Initialize Containers
        oof_preds = []
        oof_targets = []
        oof_ids = []

        test_preds_accum = None
        test_ids = None

        # Load Test Loader (Shared across folds)
        test_loader = get_test_dataloader(load_cached_data=True)

        # 3. Iterate Folds
        n_folds = 1 if self.debug else Config.N_FOLDS

        for fold_idx in range(n_folds):
            self.logger.log(f"Processing Fold {fold_idx}/{n_folds-1}...")

            # Load Validation Loader
            _, val_loader = get_dataloaders(fold_idx, load_cached_data=True)

            # Load Expert Model
            path_a = self.model_a_path_fmt.format(fold_idx)
            model_a = self._load_model(Config.MODEL_A_NAME, path_a)

            if model_a is None:
                if self.debug:
                    self.logger.log("Skipping fold due to missing model (Debug Mode).")
                    continue
                else:
                    raise FileNotFoundError(f"Missing model for fold {fold_idx}")

            # --- Inference: Validation (OOF) ---
            # Predict with TTA
            preds_a = predict_tta(model_a, val_loader, self.device)

            # Extract Targets and IDs
            fold_targets = []
            fold_ids = []
            for _, t, i in val_loader:
                fold_targets.append(t.numpy())
                fold_ids.extend(i)
            fold_targets = np.concatenate(fold_targets)

            # Store
            oof_preds.append(preds_a)
            oof_targets.append(fold_targets)
            oof_ids.extend(fold_ids)

            # --- Inference: Test ---
            t_preds_a = predict_tta(model_a, test_loader, self.device)

            if test_preds_accum is None:
                test_preds_accum = t_preds_a
                # Extract Test IDs once
                t_ids = []
                for _, _, i in test_loader:
                    t_ids.extend(i)
                test_ids = t_ids
            else:
                test_preds_accum += t_preds_a

            # Cleanup
            del model_a, preds_a, t_preds_a
            torch.cuda.empty_cache()

        # 4. Aggregate Results
        # OOF: Concatenate across folds
        data = {
            "oof_preds": np.concatenate(oof_preds, axis=0),
            "oof_targets": np.concatenate(oof_targets, axis=0),
            "oof_ids": np.array(oof_ids),
            # Test: Average across folds
            "test_preds": test_preds_accum / n_folds,
            "test_ids": np.array(test_ids),
        }

        # 5. Save Cache
        if not self.debug:
            os.makedirs(self.work_dir, exist_ok=True)
            np.save(self.cache_path, data)
            self.logger.log(f"Saved stacking data to {self.cache_path}")

        return data

    def predict_and_submit(self, data):
        """
        Generates final predictions for the test set and saves the submission file.

        Args:
            data (dict): Data dictionary containing test predictions.
        """
        self.logger.log("Generating Final Submission...")

        # Use averaged probabilities directly
        final_probs = data["test_preds"]

        # Prepare Submission DataFrame
        # Get breed names for columns
        label_map = get_label_map()
        # Invert map to get list of breeds sorted by index
        idx_to_breed = {v: k for k, v in label_map.items()}
        columns = [idx_to_breed[i] for i in range(Config.NUM_CLASSES)]

        df = pd.DataFrame(final_probs, columns=columns)

        # Insert ID column
        df.insert(0, "id", data["test_ids"])

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df.to_csv(Config.SUBMISSION_PATH, index=False)

        self.logger.log(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.log(f"Submission shape: {df.shape}")
