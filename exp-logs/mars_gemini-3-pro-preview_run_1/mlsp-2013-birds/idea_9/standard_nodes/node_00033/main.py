import sys
import os
import warnings
import numpy as np
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    STUDENT_WIDTH,
    IMG_HEIGHT,
    BATCH_SIZE,
    DEVICE,
    SEED,
    NUM_CLASSES,
)
from library.utils import set_seed, save_submission
from library.data import load_metadata, process_and_cache_data, get_loader
from library.pipeline import (
    train_teachers,
    generate_ensemble_pseudo_labels,
    train_student_with_swa,
)
from library.engine import evaluate


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Pipeline Execution

    # Stage 1: Train Teachers
    # We use the full training configuration (debug=False) to ensure maximum performance.
    # The dataset is small, so this fits well within the time limit.
    print("Starting Stage 1: Teacher Training...")
    teacher_paths = train_teachers(debug=False)

    # Stage 2: Pseudo-Label Generation
    print("Starting Stage 2: Pseudo-Label Generation...")
    test_ids_pseudo, pseudo_probs = generate_ensemble_pseudo_labels(
        teacher_paths, debug=False
    )

    # Stage 3: Student Training with SWA
    print("Starting Stage 3: Student Training with SWA...")
    student_model = train_student_with_swa(test_ids_pseudo, pseudo_probs, debug=False)

    # 3. Final Validation
    print("\n--- Final Validation ---")
    df_val = load_metadata("val")
    # Load validation data (using cache if available)
    val_images, val_labels, val_ids = process_and_cache_data(
        df_val, "val", STUDENT_WIDTH, IMG_HEIGHT, load_cached_data=True
    )
    val_loader = get_loader(val_images, val_labels, val_ids, BATCH_SIZE, shuffle=False)

    # Evaluate
    val_loss, val_auc, val_probs, val_targets = evaluate(
        student_model, val_loader, DEVICE
    )

    # Print required metric format
    print(f"Final Validation Metric: {val_auc:.16f}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Mean Absolute Error per sample across all classes
    # val_targets and val_probs are (N, 19)
    sample_errors = np.abs(val_targets - val_probs).mean(axis=1)

    # Feature 1: Label Cardinality (number of active species)
    label_cardinality = val_targets.sum(axis=1)

    # Feature 2 & 3: Image Statistics (Mean and Std)
    # val_images is a Tensor (N, 3, H, W)
    # Convert to numpy for stats
    imgs_np = val_images.numpy()
    # Mean over (Channel, Height, Width) -> (N,)
    img_means = imgs_np.mean(axis=(1, 2, 3))
    img_stds = imgs_np.std(axis=(1, 2, 3))

    # Compute Correlations
    def print_correlation(feat_name, feat_values, errors):
        if np.std(feat_values) < 1e-9:
            print(f"Correlation (Error vs {feat_name}): Undefined (Zero Variance)")
        else:
            # using numpy corrcoef
            corr = np.corrcoef(feat_values, errors)[0, 1]
            print(f"Correlation (Error vs {feat_name}): {corr:.4f}")

    print_correlation("Label Cardinality", label_cardinality, sample_errors)
    print_correlation("Image Mean Intensity", img_means, sample_errors)
    print_correlation("Image Contrast (Std)", img_stds, sample_errors)

    # 5. Submission Logic
    THRESHOLD = 0.9433543480067271

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric {val_auc:.6f} > {THRESHOLD}. Generating submission..."
        )

        # Load Test Data for Inference (Student Resolution)
        df_test = load_metadata("test")
        test_images, test_labels, test_ids = process_and_cache_data(
            df_test, "test", STUDENT_WIDTH, IMG_HEIGHT, load_cached_data=True
        )
        test_loader = get_loader(
            test_images, test_labels, test_ids, BATCH_SIZE, shuffle=False
        )

        # Predict
        _, _, test_probs, _ = evaluate(student_model, test_loader, DEVICE)

        # Save
        save_submission(test_ids, test_probs, SUBMISSION_PATH)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(f"\nValidation metric {val_auc:.6f} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
