import os
import numpy as np
import pandas as pd
import torch
from torch.nn import Sigmoid

from library.config import Config
from library.modeling import InsultModel
from library.data_factory import create_dataloaders, get_tokenizer
from library.utils import seed_everything


def generate_submission():
    """
    Generates the submission file by:
    1. Loading the test dataset.
    2. Loading the ensemble of trained models (one per seed).
    3. Running inference to get probabilities.
    4. Averaging predictions across seeds (Seed Averaging).
    5. Saving the results to the submission file.
    """
    print("Starting Inference and Submission Generation...")

    # Ensure reproducibility
    seed_everything(Config.seeds[0])

    # Create submission directory
    os.makedirs(Config.submission_dir, exist_ok=True)

    # 1. Prepare Data
    print("Preparing Test DataLoader...")
    tokenizer = get_tokenizer()
    test_dataloader = create_dataloaders(
        stage="test", tokenizer=tokenizer, load_cached_data=True
    )

    # 2. Run Inference for each Seed
    all_seed_predictions = []
    sigmoid = Sigmoid()

    for seed in Config.seeds:
        model_path = os.path.join(Config.working_dir, f"model_seed_{seed}.bin")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model for seed {seed} from {model_path}...")

        # Initialize model architecture (no need to load pretrained weights, we load state_dict)
        model = InsultModel(pretrained=False)

        # Load trained weights
        state_dict = torch.load(model_path, map_location=Config.device)
        model.load_state_dict(state_dict)

        model.to(Config.device)
        model.eval()

        seed_preds = []

        print(f"Running inference for seed {seed}...")
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)

                # Forward pass returns logits
                logits = model(input_ids, attention_mask)

                # Apply Sigmoid to get probabilities [0, 1]
                probs = sigmoid(logits).view(-1).cpu().numpy()
                seed_preds.extend(probs)

        all_seed_predictions.append(np.array(seed_preds))

        # Clean up to save memory
        del model, state_dict
        torch.cuda.empty_cache()

    if not all_seed_predictions:
        raise RuntimeError("No models were found. Cannot generate submission.")

    # 3. Ensemble (Mean Averaging)
    print(f"Ensembling predictions from {len(all_seed_predictions)} models...")
    all_seed_predictions = np.array(all_seed_predictions)
    avg_predictions = np.mean(all_seed_predictions, axis=0)

    # 4. Create Submission File
    print("Creating submission file...")

    # Load the sample submission to preserve format
    sample_sub_path = os.path.join("./input", "sample_submission_null.csv")

    try:
        submission_df = pd.read_csv(sample_sub_path)
    except Exception as e:
        print(f"Error reading sample submission: {e}")
        # Fallback: create dataframe from metadata/test.csv
        print("Falling back to metadata/test.csv structure.")
        test_df = pd.read_csv(Config.test_path)
        submission_df = test_df.copy()
        # Ensure Insult column exists
        submission_df["Insult"] = 0.0

    # Verify lengths match
    if len(submission_df) != len(avg_predictions):
        print(
            f"Warning: Length mismatch. Submission: {len(submission_df)}, Preds: {len(avg_predictions)}"
        )
        # If mismatch, we truncate or pad?
        # Usually indicates data loading issue. We assume they match based on metadata generation.
        # We will assign by index up to the minimum length to avoid crash, but this is critical.
        min_len = min(len(submission_df), len(avg_predictions))
        submission_df.iloc[:min_len, submission_df.columns.get_loc("Insult")] = (
            avg_predictions[:min_len]
        )
    else:
        submission_df["Insult"] = avg_predictions

    # Save
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print("Sample of predictions:")
    print(submission_df[["Insult"]].head())
