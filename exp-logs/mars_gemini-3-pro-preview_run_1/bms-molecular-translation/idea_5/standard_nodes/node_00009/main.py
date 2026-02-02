import sys
import os
import torch
import pandas as pd
import numpy as np
import nltk
from torch.utils.data import DataLoader

# Add current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.engine import run_training, generate_predictions, set_seed
from library.dataset import prepare_datasets, collate_fn
from library.model import AttributeAugmentedAttnNet
from library.utils import Tokenizer


def main():
    # 1. Initialize Configuration for Fast Training
    config = Config()

    # Override config for a fast baseline training run
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 12000  # Train on a subset to save time
    config.EPOCHS = 2  # Limit epochs for speed
    config.BATCH_SIZE = 64
    config.NUM_WORKERS = 2  # Reduce workers to minimize overhead

    # Ensure reproducibility
    set_seed(config.SEED)

    print("--- Starting Fast Baseline Training ---")
    # Execute training (uses DEBUG=True settings)
    run_training(config)

    # 2. Full Validation Assessment
    print("\n--- Starting Full Validation Assessment ---")

    # Switch to full dataset mode for validation requirements
    config.DEBUG = False

    # Re-load datasets. We only need the validation set here.
    # load_cached_data=True ensures we use the tokenizer fitted during training preparation
    _, val_dataset, _, tokenizer = prepare_datasets(config, load_cached_data=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,  # Increase batch size for inference speed
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load the best model saved during training
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model not found at {config.MODEL_PATH}")
        return

    model = AttributeAugmentedAttnNet(config, tokenizer.vocab_size)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    predictions = []
    ground_truths = []
    lengths = []

    print(f"Running inference on {len(val_dataset)} validation samples...")

    # Inference loop without gradient calculation
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data["image"].to(config.DEVICE)
            original_texts = data["original_text"]

            # Forward pass with greedy decoding (targets=None)
            seq_logits, _ = model(images, targets=None, teacher_forcing_ratio=0.0)

            # Decode sequences: (B, MaxLen, Vocab) -> (B, MaxLen)
            pred_indices = torch.argmax(seq_logits, dim=2)

            # Convert indices to text
            for idx in range(len(original_texts)):
                pred_text = tokenizer.sequence_to_text(pred_indices[idx])
                predictions.append(pred_text)
                ground_truths.append(original_texts[idx])
                lengths.append(len(original_texts[idx]))

    # 3. Calculate Final Metric
    distances = []
    for pred, ref in zip(predictions, ground_truths):
        d = nltk.edit_distance(pred, ref)
        distances.append(d)

    final_metric = np.mean(distances)

    # PRINT REQUIRED METRIC FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Create a DataFrame to analyze correlations
    df_analysis = pd.DataFrame(
        {"ground_truth_len": lengths, "levenshtein_error": distances}
    )

    # Calculate correlation
    correlation = df_analysis["ground_truth_len"].corr(df_analysis["levenshtein_error"])
    print(f"Correlation between Ground Truth Length and Error: {correlation}")

    # 5. Submission Generation
    # Threshold defined in requirements
    THRESHOLD = 81.60407868615773

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")

        # We need to ensure config.DEBUG is False so we predict on the full test set
        # (It was already set to False before validation)
        generate_predictions(config)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
