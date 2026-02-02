import os
import torch
import numpy as np
import scipy.stats
from library.config import Config
from library.utils import set_seed, get_device
from library.trainer import Trainer
from library.inference import Predictor


def perform_failure_analysis(trainer, device):
    """
    Analyzes model performance on the validation set to identify error correlations.
    """
    print("Performing Failure Analysis on Validation Set...")
    trainer.model.eval()

    errors = []
    seq_lengths = []
    num_gestures = []

    with torch.no_grad():
        for features, targets, mask, _ in trainer.val_loader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            # Forward Pass
            outputs = trainer.model(features, mask)
            logits = outputs["stage3"]

            for i in range(len(features)):
                # Decode
                pred_seq = trainer._decode_predictions(logits[i], mask[i])
                true_seq = trainer._decode_targets(targets[i], mask[i])

                # Calculate metrics
                dist = trainer._levenshtein_distance(pred_seq, true_seq)
                length = mask[i].sum().item()
                n_gestures = len(true_seq)

                errors.append(dist)
                seq_lengths.append(length)
                num_gestures.append(n_gestures)

    # Convert to numpy for correlation calculation
    errors = np.array(errors)
    seq_lengths = np.array(seq_lengths)
    num_gestures = np.array(num_gestures)

    # Calculate Correlations
    if len(errors) > 1:
        corr_len, _ = scipy.stats.pearsonr(errors, seq_lengths)
        corr_gest, _ = scipy.stats.pearsonr(errors, num_gestures)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Number of Gestures): {corr_gest:.4f}")
    else:
        print("Not enough samples for correlation analysis.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Training
    # We use the full dataset (limit=None) to ensure best performance within the time limit.
    # The A100 GPU is sufficient to handle this quickly.
    print("Initializing Trainer...")
    # Cite debug_lesson_4: Force reprocessing to ensure full dataset is loaded, fixing stale cache issue.
    trainer = Trainer(load_cached_data=False, limit=None)

    print("Starting Training Pipeline...")
    trainer.fit()

    # 3. Final Evaluation
    print("Loading best model for final evaluation...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=device)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    print("Computing Final Validation Metric...")
    _, final_metric = trainer.validate(epoch=0)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    perform_failure_analysis(trainer, device)

    # 5. Submission
    # Threshold defined in task description
    THRESHOLD = 0.1282225237449118

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        predictor = Predictor(model_path=Config.BEST_MODEL_PATH)
        predictor.run_inference(load_cached_data=True)
    else:
        print(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
