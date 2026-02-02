import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.model import EfficientNetExpert
from library.data import get_test_dataloader


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the Verified Content-Anchored Ensemble (VCAE).

    Process:
    1. Iterates through each Expert (A, B, C) corresponding to offsets -5, 0, +5.
    2. Loads the 5-fold ensemble models for the current Expert.
    3. Generates slice-specific predictions for all test subjects.
    4. Aggregates predictions across folds (intra-expert) and then across experts (inter-expert).
    5. Saves the final submission to CSV.

    Args:
        load_cached_data (bool): If True, uses cached Center of Mass data.
                                 If False, recomputes CoM from DICOM volumes.
    """
    device = torch.device(Config.DEVICE)
    print(f"Starting Inference on device: {device}")

    # Container for aggregating predictions: {subject_id: [prob_expert_A, prob_expert_B, prob_expert_C]}
    subject_predictions = {}

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Iterate through each Expert defined in the VCAE strategy
    for expert_name, offset in Config.EXPERTS.items():
        print(f"\nProcessing {expert_name} (Offset: {offset})...")

        # 1. Load Ensemble Models for this Expert
        expert_models = []
        for fold in range(Config.NUM_FOLDS):
            model_path = os.path.join(
                Config.WORK_DIR, f"best_model_{expert_name}_fold{fold}.pth"
            )

            # Check if model exists (robustness for partial training runs)
            if not os.path.exists(model_path):
                print(
                    f"  [Warning] Model not found: {model_path}. Skipping fold {fold}."
                )
                continue

            try:
                model = EfficientNetExpert(pretrained=False)  # Architecture only
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                expert_models.append(model)
            except Exception as e:
                print(f"  [Error] Failed to load {model_path}: {e}")

        if not expert_models:
            print(
                f"  [Error] No valid models found for {expert_name}. Skipping this expert."
            )
            continue

        print(f"  Loaded {len(expert_models)} models for {expert_name}.")

        # 2. Get Data Loader for this Expert's specific anatomical view
        # This handles CoM calculation and caching internally via library.data/utils
        test_loader = get_test_dataloader(
            expert_offset=offset, load_cached_data=load_cached_data
        )

        # 3. Inference Loop
        with torch.no_grad():
            for images, subject_ids in test_loader:
                images = images.to(device)

                # Collect predictions from all folds for this batch
                fold_probs = []
                for model in expert_models:
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    fold_probs.append(probs.cpu().numpy())

                # Average across folds -> (Batch_Size, 1)
                # Shape of fold_probs: (Num_Models, Batch_Size, 1)
                avg_expert_probs = np.mean(fold_probs, axis=0).flatten()

                # Store results
                for sid, prob in zip(subject_ids, avg_expert_probs):
                    # Convert sid to int for consistency
                    sid = int(sid)
                    if sid not in subject_predictions:
                        subject_predictions[sid] = []
                    subject_predictions[sid].append(prob)

    # 4. Final Aggregation and Submission Generation
    print("\nAggregating predictions and generating submission...")

    final_rows = []

    # Sort by ID for clean output
    sorted_ids = sorted(subject_predictions.keys())

    for sid in sorted_ids:
        probs = subject_predictions[sid]

        if not probs:
            # Fallback for completely failed subjects (should not happen)
            final_prob = 0.5
        else:
            # Average the probabilities from the available Experts (A, B, C)
            final_prob = np.mean(probs)

        final_rows.append({"BraTS21ID": sid, "MGMT_value": final_prob})

    # Create DataFrame
    submission_df = pd.DataFrame(final_rows)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(submission_df.head())
