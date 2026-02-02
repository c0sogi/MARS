import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.dataset import SoADataset, SoACollator
from library.model import DMPNN
from library.losses import LossComputer


class Predictor:
    """
    Handles the inference phase for the Scalar Coupling Prediction task.
    Loads a trained model, processes the test dataset, and generates a submission CSV.
    """

    def __init__(self):
        self.config = Config
        self.device = torch.device(self.config.DEVICE)

        # Ensure reproducibility
        self.config.set_seed(self.config.SEED)

        # Initialize Model Architecture
        # We must use the same hyperparameters as training to match the state dict
        self.model = DMPNN(
            hidden_dim=self.config.HIDDEN_DIM,
            num_layers=self.config.NUM_LAYERS,
            num_rbf=self.config.NUM_RBF,
            num_angle_rbf=self.config.NUM_ANGLE_RBF,
            rbf_gamma=self.config.RBF_GAMMA,
            dropout=self.config.DROPOUT,
        ).to(self.device)

        # Initialize LossComputer
        # This is used primarily for its 'unstandardize' method and loaded statistics
        self.loss_computer = LossComputer().to(self.device)

    def _to_device(self, batch):
        """Moves a dictionary batch to the configured device."""
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = v.to(self.device, non_blocking=True)
            else:
                new_batch[k] = v
        return new_batch

    def predict(self, batch_size=None, num_workers=None):
        """
        Generates predictions for the test set and saves them to submission.csv.

        Args:
            batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
            num_workers (int, optional): Number of workers for data loading. Defaults to Config.NUM_WORKERS.
        """
        # Use config defaults if arguments not provided
        batch_size = batch_size if batch_size is not None else self.config.BATCH_SIZE
        num_workers = (
            num_workers if num_workers is not None else self.config.NUM_WORKERS
        )

        print("Starting inference process...")

        # 1. Load Test Dataset
        # We rely on the cached processed data. If it doesn't exist, SoADataset will raise an error.
        print("Loading test dataset...")
        try:
            test_dataset = SoADataset(split="test", load_cached_data=True)
        except RuntimeError as e:
            print(f"Error loading test dataset: {e}")
            print(
                "Ensure that the DataProcessor has been run and test data is processed."
            )
            return

        test_collator = SoACollator(test_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=test_collator,
            pin_memory=True,
        )

        # 2. Load Trained Model
        best_model_path = os.path.join(self.config.CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Best model checkpoint not found at {best_model_path}. "
                "Please train the model before running inference."
            )

        print(f"Loading model weights from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # 3. Inference Loop
        all_ids = []
        all_preds = []

        print(f"Running inference on {len(test_dataset)} molecules...")

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = self._to_device(batch)

                # Mixed Precision Inference
                with autocast():
                    predictions = self.model(batch)
                    # We only care about the coupling prediction for submission
                    # Shape: (num_couplings_in_batch, )
                    pred_coupling_std = predictions["coupling"]

                # 4. Inverse Transformation
                # The model predicts standardized values (z-scores).
                # We must convert them back to physical units using the per-type stats.
                coupling_types = batch["coupling_type"]

                # loss_computer.unstandardize handles the lookup of mean/std by type
                pred_coupling_phys = self.loss_computer.unstandardize(
                    pred_coupling_std, coupling_types
                )

                # 5. Collect Results
                # coupling_id is the 'id' column from sample_submission.csv
                batch_ids = batch["coupling_id"].cpu().numpy()
                batch_preds = pred_coupling_phys.float().cpu().numpy()

                all_ids.append(batch_ids)
                all_preds.append(batch_preds)

        # 6. Aggregate
        final_ids = np.concatenate(all_ids)
        final_preds = np.concatenate(all_preds)

        # 7. Create Submission DataFrame
        df_sub = pd.DataFrame(
            {"id": final_ids, "scalar_coupling_constant": final_preds}
        )

        # Ensure sorted by ID (standard practice for Kaggle-style submissions)
        df_sub = df_sub.sort_values("id")

        # 8. Save
        # Ensure submission directory exists
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)

        print(f"Saving submission file to {self.config.SUBMISSION_PATH}...")
        df_sub.to_csv(self.config.SUBMISSION_PATH, index=False)

        print("Inference complete.")
