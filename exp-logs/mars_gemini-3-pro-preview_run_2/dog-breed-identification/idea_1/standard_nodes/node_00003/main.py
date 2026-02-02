import numpy as np
import sys
from library.config import SUBMISSION_PATH, SEED, NUM_CLASSES
from library.utils import set_seed, save_submission
from library.data_loader import create_dataloaders
from library.classifier import DeepClassifier


def main():
    # 1. Setup
    set_seed(SEED)
    print("Starting execution...")

    # Baseline metric from Linear Probing (Cite solution_lesson_node_00002)
    BASELINE_METRIC = 0.5594493322384437

    # 2. Data Loading
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        debug_limit=None
    )

    # 3. Model & Training
    # We move from Linear Probing to End-to-End Fine-Tuning to allow the backbone
    # to adapt to the specific features of dog breeds.
    classifier = DeepClassifier(num_classes=len(classes))

    classifier.train(train_loader, val_loader)

    # 4. Validation
    val_loss, val_probs, y_val = classifier.evaluate(val_loader)
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample Log Loss
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    sample_indices = np.arange(len(y_val))
    true_class_probs = val_probs_clipped[sample_indices, y_val.astype(int)]
    per_sample_loss = -np.log(true_class_probs)

    # Analyze correlation with prediction confidence (Max Probability)
    max_probs = np.max(val_probs, axis=1)

    if len(per_sample_loss) > 1:
        corr_conf = np.corrcoef(per_sample_loss, max_probs)[0, 1]
        print(
            f"Correlation between Error Magnitude and Prediction Confidence: {corr_conf:.6f}"
        )

    # 6. Submission
    if val_loss < BASELINE_METRIC:
        print(
            f"\nValidation metric {val_loss:.4f} improved over baseline {BASELINE_METRIC:.4f}."
        )
        print("Generating submission...")
        test_probs, test_ids = classifier.predict(test_loader)
        save_submission(test_probs, test_ids, classes, SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {val_loss:.4f} did not improve over baseline {BASELINE_METRIC:.4f}. Skipping submission."
        )

    print("Run complete.")


if __name__ == "__main__":
    main()
