import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.data import get_test_dataloaders
from library.models import get_cnn_model, SymbolicMLP
from library.utils import seed_everything


class EnsemblePredictor:
    """
    Orchestrates the inference process using the Hybrid Neuro-Symbolic Ensemble strategy.
    Aggregates predictions from multiple CNN snapshots and MLP models across all folds.
    """

    def __init__(self, load_cached_data=True):
        """
        Args:
            load_cached_data (bool): Whether to use cached data for dataloaders.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_loader, self.mlp_loader = get_test_dataloaders(
            load_cached_data=load_cached_data
        )

        # Dictionary to store accumulated predictions
        # Key: rec_id (int)
        # Value: {'sum': np.array(shape=(19,)), 'count': int}
        self.results = {}

    def _parse_auc_from_filename(self, filepath):
        """
        Extracts the AUC score from the checkpoint filename.
        Format expected: {model_name}_fold{fold}_ep{epoch}_auc{score}.pth
        """
        match = re.search(r"auc(\d+\.\d+)", filepath)
        if match:
            return float(match.group(1))
        return 0.0

    def _get_checkpoints(self, model_name, fold_idx, top_k=None):
        """
        Retrieves checkpoint paths for a given model and fold, sorted by AUC descending.
        """
        if model_name == "mlp":
            ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, "mlp")
        else:
            ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, model_name)

        # Pattern matches: {model_name}_fold{fold_idx}_*.pth
        pattern = os.path.join(ckpt_dir, f"{model_name}_fold{fold_idx}_*.pth")
        files = glob.glob(pattern)

        if not files:
            return []

        # Sort files by AUC score in filename
        files_with_scores = []
        for f in files:
            score = self._parse_auc_from_filename(f)
            files_with_scores.append((score, f))

        # Sort descending
        files_with_scores.sort(key=lambda x: x[0], reverse=True)

        sorted_files = [f[1] for f in files_with_scores]

        if top_k is not None:
            return sorted_files[:top_k]
        return sorted_files

    def _accumulate_predictions(self, rec_ids, probs):
        """
        Updates the results dictionary with batch predictions.
        """
        for rid, p in zip(rec_ids, probs):
            # Ensure rid is a standard python int
            rid = int(rid)
            if rid not in self.results:
                self.results[rid] = {
                    "sum": np.zeros(Config.NUM_CLASSES, dtype=np.float32),
                    "count": 0,
                }
            self.results[rid]["sum"] += p
            self.results[rid]["count"] += 1

    def run_cnn_inference(self):
        """
        Runs inference for all CNN architectures.
        Strategy: Use Top-3 snapshots per fold for each architecture.
        """
        print("Starting CNN Ensemble Inference...")
        for model_name in Config.CNN_MODELS:
            for fold_idx in range(Config.N_FOLDS):
                # Get Top-3 snapshots
                checkpoints = self._get_checkpoints(model_name, fold_idx, top_k=3)

                if not checkpoints:
                    print(
                        f"Warning: No checkpoints found for {model_name} fold {fold_idx}"
                    )
                    continue

                for ckpt_path in checkpoints:
                    # Load Model
                    model = get_cnn_model(model_name, pretrained=False).to(self.device)
                    try:
                        model.load_state_dict(
                            torch.load(ckpt_path, map_location=self.device)
                        )
                    except Exception as e:
                        print(f"Error loading {ckpt_path}: {e}")
                        continue

                    model.eval()

                    # Inference
                    with torch.no_grad():
                        for batch in self.cnn_loader:
                            imgs = batch["image"].to(self.device)
                            rec_ids = batch["rec_id"].numpy()

                            outputs = model(imgs)
                            probs = torch.sigmoid(outputs).cpu().numpy()

                            self._accumulate_predictions(rec_ids, probs)

                    # Cleanup
                    del model
                    torch.cuda.empty_cache()

    def run_mlp_inference(self):
        """
        Runs inference for the Symbolic MLP.
        Strategy: Use Top-1 (Best) snapshot per fold.
        """
        print("Starting MLP Ensemble Inference...")
        for fold_idx in range(Config.N_FOLDS):
            # Get Top-1 snapshot
            checkpoints = self._get_checkpoints("mlp", fold_idx, top_k=1)

            if not checkpoints:
                print(f"Warning: No checkpoints found for MLP fold {fold_idx}")
                continue

            for ckpt_path in checkpoints:
                # Load Model
                model = SymbolicMLP().to(self.device)
                try:
                    model.load_state_dict(
                        torch.load(ckpt_path, map_location=self.device)
                    )
                except Exception as e:
                    print(f"Error loading {ckpt_path}: {e}")
                    continue

                model.eval()

                # Inference
                with torch.no_grad():
                    for batch in self.mlp_loader:
                        feats = batch["features"].to(self.device)
                        rec_ids = batch["rec_id"].numpy()

                        outputs = model(feats)
                        probs = torch.sigmoid(outputs).cpu().numpy()

                        self._accumulate_predictions(rec_ids, probs)

                # Cleanup
                del model
                torch.cuda.empty_cache()

    def generate_submission(self):
        """
        Averages predictions and writes the submission CSV.
        """
        print("Generating submission file...")
        final_rows = []

        # Ensure we cover all test IDs from the metadata
        test_df = pd.read_csv(Config.TEST_CSV)
        all_test_ids = test_df["rec_id"].unique()

        for rid in all_test_ids:
            rid = int(rid)
            if rid in self.results and self.results[rid]["count"] > 0:
                avg_probs = self.results[rid]["sum"] / self.results[rid]["count"]
            else:
                # Fallback (should not happen if inference ran correctly)
                avg_probs = np.zeros(Config.NUM_CLASSES)

            # Format: Id = rec_id * 100 + species_idx
            for species_idx in range(Config.NUM_CLASSES):
                row_id = rid * 100 + species_idx
                prob = float(avg_probs[species_idx])
                final_rows.append({"Id": int(row_id), "Probability": prob})

        submission_df = pd.DataFrame(final_rows)
        # Sort by Id for consistency
        submission_df = submission_df.sort_values("Id")

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

    def predict(self):
        """
        Main execution method.
        """
        seed_everything(Config.SEED)

        # 1. Run CNN Stream
        self.run_cnn_inference()

        # 2. Run MLP Stream
        self.run_mlp_inference()

        # 3. Generate Output
        self.generate_submission()


def run_inference(load_cached_data=True):
    """
    Module entry point for running inference.

    Args:
        load_cached_data (bool): Whether to use cached data artifacts.
    """
    predictor = EnsemblePredictor(load_cached_data=load_cached_data)
    predictor.predict()
