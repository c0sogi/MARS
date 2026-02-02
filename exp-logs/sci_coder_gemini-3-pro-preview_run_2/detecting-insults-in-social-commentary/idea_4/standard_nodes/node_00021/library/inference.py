import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import seed_everything
from library.dataset import InsultDataset, get_tokenizer, load_dataset_dataframe
from library.model import InsultModel


def predict_fn(
    test_path=Config.test_path,
    submission_input_path=os.path.join(Config.input_dir, "sample_submission_null.csv"),
    submission_output_path=Config.submission_path,
    model_dir=Config.working_dir,
    batch_size=Config.batch_size * 2,
    num_workers=Config.num_workers,
    device=Config.device,
    use_fp16=Config.use_fp16,
):
    """
    Executes the inference pipeline for the Insult Detection task.
    Aggregates predictions from all trained models in the heterogeneous ensemble.

    Args:
        test_path (str): Path to the test metadata CSV.
        submission_input_path (str): Path to the sample submission file.
        submission_output_path (str): Path to save the final submission.
        model_dir (str): Directory containing trained model weights.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of dataloader workers.
        device (torch.device): Device to run inference on.
        use_fp16 (bool): Whether to use mixed precision.
    """
    seed_everything(Config.seed)

    # 1. Load Test Data
    # We use the caching mechanism provided in library.dataset
    print(f"Loading test data from {test_path}...")
    test_df = load_dataset_dataframe(test_path, "test_cleaned", load_cached_data=True)

    # 2. Prepare for Ensemble Prediction
    # Initialize an array to accumulate probabilities
    total_preds = np.zeros(len(test_df))
    model_count = 0

    # 3. Iterate over all Backbones and Folds
    for model_name in Config.model_backbones:
        print(f"\nProcessing Backbone: {model_name}")

        # Load Tokenizer for this backbone
        tokenizer = get_tokenizer(model_name)

        # Create Dataset and Loader
        # We create the dataset once per backbone as tokenization depends on the backbone
        test_dataset = InsultDataset(test_df, tokenizer, Config.max_len, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        model_name_safe = model_name.replace("/", "_")

        for fold in range(Config.n_folds):
            weight_path = os.path.join(model_dir, f"{model_name_safe}_fold_{fold}.pth")

            if not os.path.exists(weight_path):
                print(f"Warning: Model weight not found at {weight_path}. Skipping...")
                continue

            print(f"  Predicting with Fold {fold}...")

            # Initialize Model
            model = InsultModel(model_name, Config)
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            fold_preds = []

            # Inference Loop
            with torch.no_grad():
                for data in test_loader:
                    input_ids = data["input_ids"].to(device)
                    attention_mask = data["attention_mask"].to(device)

                    with autocast(enabled=use_fp16):
                        outputs = model(input_ids, attention_mask)
                        # Apply sigmoid to get probabilities [0, 1]
                        probs = torch.sigmoid(outputs).view(-1)

                    fold_preds.append(probs.cpu().numpy())

            # Accumulate predictions
            total_preds += np.concatenate(fold_preds)
            model_count += 1

            # Cleanup to save memory
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # 4. Average Predictions
    if model_count == 0:
        raise RuntimeError(
            "No models were found or executed. Cannot generate submission."
        )

    avg_preds = total_preds / model_count
    print(f"\nEnsemble prediction complete. Averaged over {model_count} models.")

    # 5. Generate Submission File
    print(f"Generating submission file at {submission_output_path}...")

    # Load sample submission to ensure correct format
    if os.path.exists(submission_input_path):
        submission = pd.read_csv(submission_input_path)
        # Handle potential index columns in sample file
        if "Unnamed: 0" in submission.columns:
            submission = submission.drop(columns=["Unnamed: 0"])
    else:
        # Fallback if sample submission is missing, create from test_df
        print("Sample submission not found. Creating from test dataframe.")
        submission = test_df.copy()
        if "Insult" not in submission.columns:
            submission["Insult"] = 0

    # Ensure lengths match
    if len(submission) != len(avg_preds):
        print(
            f"Warning: Submission length ({len(submission)}) does not match prediction length ({len(avg_preds)}). Truncating or aligning."
        )
        # In a strict competition setting, this might raise an error, but here we align to predictions
        submission = submission.iloc[: len(avg_preds)]

    # Assign predictions
    submission["Insult"] = avg_preds

    # Save
    os.makedirs(os.path.dirname(submission_output_path), exist_ok=True)
    submission.to_csv(submission_output_path, index=False)
    print("Submission saved successfully.")
