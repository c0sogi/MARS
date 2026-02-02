import sys
import os
import pandas as pd
import torch
import nltk
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Ensure library modules are importable
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_levenshtein
from library.dataset import ChemicalDataset, get_transforms
from library.train import train_model
from library.predict import generate_submission
from library.tokenizer import InchiTokenizer


def run_pipeline():
    # --- 1. Configuration for Fast Baseline ---
    print("--- Configuring Pipeline ---")
    # Limit training to 1 epoch and 50k samples to fit within the 2-hour runtime constraint
    # while still providing a meaningful baseline.
    Config.NUM_EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50000

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # --- 2. Model Training ---
    print("\n--- Starting Training Phase ---")
    # train_model will use the DEBUG_SIZE set above for the training set
    model = train_model(debug=Config.DEBUG, epochs=Config.NUM_EPOCHS)

    # --- 3. Full Validation Assessment ---
    print("\n--- Starting Full Validation Assessment ---")
    device = Config.DEVICE
    tokenizer = InchiTokenizer()

    # Load the COMPLETE validation set (ignoring DEBUG flag used for training)
    print(f"Loading full validation metadata from {Config.VAL_METADATA_PATH}...")
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    val_dataset = ChemicalDataset(
        val_df, tokenizer, transform=get_transforms("val"), mode="val"
    )

    # Use a larger batch size for inference if memory allows, or stick to Config.BATCH_SIZE
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()

    predictions = []
    targets = []
    target_lengths = []

    print(f"Evaluating on {len(val_dataset)} validation samples...")

    with torch.no_grad():
        for i, (images, captions) in enumerate(val_loader):
            images = images.to(device)

            # Encoder Forward
            h, c = model.encoder(images)

            # Decoder Inference (Greedy Search)
            # Initialize with <SOS>
            start_token = torch.full(
                (images.size(0), 1),
                tokenizer.sos_token_id,
                dtype=torch.long,
                device=device,
            )
            inputs = start_token

            batch_preds_indices = []

            for _ in range(Config.MAX_PRED_LEN):
                output_logits, (h, c) = model.decoder(inputs, h, c)
                # Greedy selection
                predicted_token = output_logits.argmax(dim=2)
                batch_preds_indices.append(predicted_token)
                # Next input
                inputs = predicted_token

            # Concatenate sequence
            batch_preds_indices = torch.cat(batch_preds_indices, dim=1)

            # Decode to Text
            batch_pred_strs = [
                tokenizer.sequence_to_text(seq) for seq in batch_preds_indices
            ]
            batch_target_strs = [tokenizer.sequence_to_text(seq) for seq in captions]

            predictions.extend(batch_pred_strs)
            targets.extend(batch_target_strs)

            # Store lengths for failure analysis
            for t in batch_target_strs:
                target_lengths.append(len(t))

            if (i + 1) % 100 == 0:
                print(f"Validated {i + 1}/{len(val_loader)} batches...")

    # --- 4. Metric Computation ---
    # Compute mean Levenshtein distance on the entire validation set
    final_metric = compute_levenshtein(predictions, targets)
    print(f"Final Validation Metric: {final_metric}")

    # --- 5. Failure Analysis ---
    print("\n--- Performing Failure Analysis ---")
    # Calculate individual errors
    errors = [nltk.edit_distance(p, t) for p, t in zip(predictions, targets)]

    # Calculate correlation between Error Magnitude and Target Sequence Length
    # This helps identify if the model fails on more complex (longer) molecules
    corr_len, _ = pearsonr(errors, target_lengths)

    print(f"Correlation (Levenshtein Error vs InChI Length): {corr_len:.6f}")
    if corr_len > 0.3:
        print(
            "Analysis: Strong positive correlation detected. The model performance degrades significantly as molecule complexity increases."
        )
    elif corr_len > 0.1:
        print("Analysis: Weak positive correlation detected.")
    else:
        print("Analysis: No significant correlation with sequence length.")

    # --- 6. Submission Generation ---
    print("\n--- Generating Submission for Test Set ---")
    # Generate submission using the trained model on the full test set (debug=False)
    generate_submission(model=model, debug=False)

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    run_pipeline()
