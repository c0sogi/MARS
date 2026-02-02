import os
import sys
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.config import Config
from library import utils, data, training, distillation, model


def main():
    # 1. Setup
    utils.set_seed(Config.SEED)
    logger = utils.get_logger("training.log")
    device = utils.get_device()

    logger.info(f"Starting pipeline on device: {device}")

    # 2. Stage 1: Train Teacher Ensemble
    teachers = []
    logger.info("=== Stage 1: Training Teacher Ensemble ===")

    # We use the standard dataloaders (only labeled train data)
    # We retrieve them once here, but the trainer will iterate over them.
    # Note: data.get_dataloaders returns new loaders each time called.

    for i in range(Config.NUM_TEACHERS):
        logger.info(f"--- Training Teacher {i+1}/{Config.NUM_TEACHERS} ---")
        train_loader, val_loader, _ = data.get_dataloaders()

        trainer = training.Trainer(train_loader, val_loader, device, logger)
        best_teacher, best_auc = trainer.run(save_name=f"teacher_{i}")
        teachers.append(best_teacher)

        # Clear memory
        del trainer
        torch.cuda.empty_cache()

    # 3. Stage 2: Distillation Generation 1 (Ensemble -> Student 1)
    logger.info("=== Stage 2: Distillation Generation 1 (Ensemble -> Student 1) ===")

    # Generate pseudo-labels using the teacher ensemble
    logger.info("Generating pseudo-labels with Teacher Ensemble...")
    pseudo_labels_g1 = distillation.generate_pseudo_labels(teachers, device)

    # Get dataloaders with combined data (Labeled + Pseudo-Labeled)
    train_loader_s1, val_loader_s1, _ = data.get_dataloaders(
        pseudo_labels_df=pseudo_labels_g1
    )

    logger.info("Training Student 1...")
    trainer_s1 = training.Trainer(train_loader_s1, val_loader_s1, device, logger)
    student_1, s1_auc = trainer_s1.run(save_name="student_1")

    # Clear memory
    del trainer_s1, teachers
    torch.cuda.empty_cache()

    # 4. Stage 3: Distillation Generation 2 (Student 1 -> Student 2)
    logger.info("=== Stage 3: Distillation Generation 2 (Student 1 -> Student 2) ===")

    # Regenerate pseudo-labels using Student 1
    logger.info("Refining pseudo-labels with Student 1...")
    pseudo_labels_g2 = distillation.generate_pseudo_labels(student_1, device)

    # Get dataloaders with refined combined data
    train_loader_s2, val_loader_s2, _ = data.get_dataloaders(
        pseudo_labels_df=pseudo_labels_g2
    )

    logger.info("Training Student 2...")
    trainer_s2 = training.Trainer(train_loader_s2, val_loader_s2, device, logger)
    student_2, final_auc = trainer_s2.run(save_name="student_2")

    # 5. Validation & Failure Analysis
    logger.info("=== Validation & Failure Analysis ===")

    # We perform a dedicated validation pass to get per-sample predictions for analysis
    # Use a clean validation loader (just in case)
    _, val_loader_clean, _ = data.get_dataloaders()

    student_2.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in val_loader_clean:
            images = images.to(device)
            # Forward pass
            outputs = student_2(images)
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Compute Final Metric
    final_metric = utils.calculate_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Label Cardinality
    # Error defined as Mean Absolute Error per sample across all classes
    # Cardinality defined as number of active labels per sample
    mae_per_sample = np.mean(np.abs(all_targets - all_preds), axis=1)
    cardinality_per_sample = np.sum(all_targets, axis=1)

    # Handle case where cardinality might be constant (unlikely but possible in tiny subsets)
    if np.std(cardinality_per_sample) > 0:
        correlation = np.corrcoef(mae_per_sample, cardinality_per_sample)[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error and Label Cardinality: {correlation}")

    # 6. Submission
    threshold = 0.9594082190886809
    if final_metric > threshold:
        logger.info(
            f"Metric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Generate predictions for Test Set using Student 2 (with TTA enabled in generate_pseudo_labels)
        # generate_pseudo_labels returns a DataFrame with [rec_id, species_0, ..., species_18]
        # This function applies TTA internally.
        test_preds_df = distillation.generate_pseudo_labels(student_2, device)

        # Format for submission: Id, Probability
        # Id = rec_id * 100 + species_id
        submission_data = []

        for _, row in test_preds_df.iterrows():
            rec_id = int(row["rec_id"])
            for species_idx in range(Config.NUM_CLASSES):
                prob = row[f"species_{species_idx}"]
                submission_id = rec_id * 100 + species_idx
                submission_data.append([submission_id, prob])

        submission_df = pd.DataFrame(submission_data, columns=["Id", "Probability"])

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Metric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
