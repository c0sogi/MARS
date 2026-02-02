import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_map5
from library.dataset import WhaleDataset, get_transforms, get_class_mapping
from library.model import WhaleDenseNet
from library.engine import train_model


def run_ensemble_inference(loader, models, device):
    """
    Runs inference using an ensemble of models with Test-Time Augmentation (Horizontal Flip).
    Returns averaged logits and targets (if available).
    """
    all_logits = []
    all_targets = []
    all_filenames = []

    # Ensure models are in eval mode
    for model in models:
        model.eval()

    with torch.no_grad():
        for batch in loader:
            # Handle different return types from dataset (test returns filename, val returns label)
            if len(batch) == 2:
                images, targets_or_filenames = batch
            else:
                raise ValueError("Unexpected batch structure")

            images = images.to(device)

            batch_logits_sum = None

            # Iterate over each model in the ensemble
            for model in models:
                # 1. Original View
                logits = model(images, labels=None)

                # 2. Flipped View (TTA)
                images_flip = torch.flip(images, [3])
                logits_flip = model(images_flip, labels=None)

                # Average views for this model
                model_avg_logits = (logits + logits_flip) / 2.0

                if batch_logits_sum is None:
                    batch_logits_sum = model_avg_logits
                else:
                    batch_logits_sum += model_avg_logits

            # Average across ensemble
            final_batch_logits = batch_logits_sum / len(models)

            all_logits.append(final_batch_logits.cpu())

            if isinstance(targets_or_filenames, torch.Tensor):
                all_targets.append(targets_or_filenames.cpu())
            else:
                all_filenames.extend(targets_or_filenames)

    # Concatenate results
    all_logits = torch.cat(all_logits, dim=0)

    if len(all_targets) > 0:
        all_targets = torch.cat(all_targets, dim=0)
        return all_logits, all_targets
    else:
        return all_logits, all_filenames


def perform_failure_analysis(val_logits, val_targets, class_to_idx):
    """
    Analyzes validation errors.
    Calculates correlations between error magnitude and:
    1. Class Frequency (in training set)
    2. Prediction Confidence
    """
    print("\n==== Failure Analysis ====")

    # 1. Calculate Per-Sample MAP@5 (Metric)
    # calculate_map5 usually averages, but we need per-sample here.
    # We'll reimplement the logic for per-sample scores.

    probs = F.softmax(val_logits, dim=1)
    confidences, _ = probs.max(dim=1)

    _, top5_indices = val_logits.topk(5, dim=1, largest=True, sorted=True)

    scores = []
    val_targets_np = val_targets.numpy()
    top5_indices_np = top5_indices.numpy()

    for i in range(len(val_targets)):
        t = val_targets_np[i]
        p = top5_indices_np[i]
        score = 0.0
        for k in range(min(5, len(p))):
            if p[k] == t:
                score = 1.0 / (k + 1)
                break
        scores.append(score)

    scores = np.array(scores)
    errors = 1.0 - scores  # Error magnitude (0 = perfect, 1 = complete fail)

    # 2. Get Class Frequencies from Training Data
    df_train = pd.read_csv(Config.TRAIN_CSV)
    class_counts = df_train["Id"].value_counts().to_dict()

    # Map targets to class names then to counts
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_counts = []
    for t in val_targets_np:
        class_name = idx_to_class[t]
        target_counts.append(class_counts.get(class_name, 0))

    target_counts = np.array(target_counts)
    confidences = confidences.numpy()

    # 3. Calculate Correlations
    # We check correlation between Error and (Frequency, Confidence)
    df_analysis = pd.DataFrame(
        {"Error": errors, "ClassFrequency": target_counts, "Confidence": confidences}
    )

    corr_freq = df_analysis["Error"].corr(df_analysis["ClassFrequency"])
    corr_conf = df_analysis["Error"].corr(df_analysis["Confidence"])

    print(f"Correlation between Error and Class Frequency: {corr_freq:.4f}")
    print(f"Correlation between Error and Prediction Confidence: {corr_conf:.4f}")

    # Simple insight
    if corr_freq < -0.1:
        print("Insight: The model performs significantly better on frequent classes.")
    if corr_conf < -0.1:
        print("Insight: The model's confidence is a reliable indicator of correctness.")


def main():
    # --------------------------------------------------------------------------
    # 0. Configuration Overrides for Fast Baseline
    # --------------------------------------------------------------------------
    # Limit epochs to ensure completion within time limits while maintaining performance
    # Increased to 25 for single model convergence (Cite solution_lesson_node_00026)
    Config.EPOCHS = 25
    Config.PATIENCE = 5

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 1. Training Phase (Ensemble Members)
    # --------------------------------------------------------------------------
    trained_models = []

    for seed in Config.SEEDS:
        print(f"\n--- Training Ensemble Member (Seed {seed}) ---")

        # Train the model (library.engine handles saving checkpoints)
        # Note: We use the imported train_model function
        best_map5 = train_model(seed)

        # Load the best checkpoint for this seed
        checkpoint_path = os.path.join(
            Config.WORKING_DIR, f"seed_{seed}", "model_best.pth.tar"
        )

        model = WhaleDenseNet(
            pretrained=False
        )  # Pretrained weights not needed for loading state_dict
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        trained_models.append(model)

    # --------------------------------------------------------------------------
    # 2. Validation Phase (Ensemble Inference)
    # --------------------------------------------------------------------------
    print("\n--- Running Ensemble Validation ---")

    val_dataset = WhaleDataset(mode="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run inference
    val_logits, val_targets = run_ensemble_inference(val_loader, trained_models, device)

    # Calculate Metric
    final_val_metric = calculate_map5(val_logits, val_targets)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_metric}")

    # --------------------------------------------------------------------------
    # 3. Failure Analysis
    # --------------------------------------------------------------------------
    # Load class mapping
    class_to_idx, _ = get_class_mapping()
    perform_failure_analysis(val_logits, val_targets, class_to_idx)

    # --------------------------------------------------------------------------
    # 4. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.6545824094604581

    if final_val_metric > THRESHOLD:
        print("\nMetric exceeds threshold. Generating submission...")

        test_dataset = WhaleDataset(mode="test", transform=get_transforms("test"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run inference on test set
        test_logits, test_filenames = run_ensemble_inference(
            test_loader, trained_models, device
        )

        # Get Top 5 Predictions
        _, top5_indices = test_logits.topk(5, dim=1, largest=True, sorted=True)
        top5_indices = top5_indices.numpy()

        # Decode Labels
        # Invert class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        submission_rows = []
        for i, filename in enumerate(test_filenames):
            indices = top5_indices[i]
            labels = [idx_to_class[idx] for idx in indices]
            label_str = " ".join(labels)
            submission_rows.append({"Image": filename, "Id": label_str})

        # Create DataFrame and Save
        df_submission = pd.DataFrame(submission_rows)
        df_submission.to_csv(Config.SUBMISSION_FILE, index=False)

        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric {final_val_metric} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
