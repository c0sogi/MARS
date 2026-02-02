import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import seed_everything
from library.data import get_test_data, Collate
from library.modeling import EssayScorer


def generate_predictions(model_paths=None, load_cached_data=True):
    """
    Generates predictions for the test set using an ensemble of trained models.

    Args:
        model_paths (list, optional): List of paths to model checkpoints.
                                      If None, defaults to the 5 folds in Config.output_dir.
        load_cached_data (bool): Whether to load pre-processed test data from cache.

    Returns:
        str: Path to the generated submission file.
    """
    # 1. Setup Environment
    seed_everything(Config.seed)
    print("=== Starting Inference ===")

    # Define default model paths if not provided
    if model_paths is None:
        model_paths = [
            os.path.join(Config.output_dir, f"model_fold_{i}.pth")
            for i in range(Config.num_folds)
        ]

    # Validate model paths
    valid_model_paths = [p for p in model_paths if os.path.exists(p)]
    if not valid_model_paths:
        # In a real scenario, we might raise an error, but for robustness we check
        print(
            f"Warning: No models found at specified paths. Checking {Config.output_dir}..."
        )
        valid_model_paths = []
        # Fallback logic could go here, but we assume models exist for this task
        if not valid_model_paths:
            raise FileNotFoundError(
                f"No model checkpoints found. Expected paths like: {model_paths}"
            )

    print(f"Found {len(valid_model_paths)} models for ensemble.")

    # 2. Load Data
    # We use the library function which handles caching logic internally
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    test_dataset = get_test_data(tokenizer, load_cached_data=load_cached_data)
    collate_fn = Collate(tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Ensemble Prediction Loop
    # Initialize array to store sum of predictions
    final_preds = np.zeros(len(test_dataset))

    for i, path in enumerate(valid_model_paths):
        print(f"Predicting with model {i+1}/{len(valid_model_paths)}: {path}")

        # Initialize Model Architecture
        # pretrained=False because we are loading custom weights
        model = EssayScorer(model_name_or_path=Config.model_name, pretrained=False)

        # Load Weights
        state_dict = torch.load(path, map_location=Config.device)
        model.load_state_dict(state_dict)
        model.to(Config.device)
        model.eval()

        fold_preds = []

        # Inference
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)

                # Mixed Precision Inference
                with autocast(enabled=Config.use_fp16):
                    outputs = model(input_ids, attention_mask)

                fold_preds.append(outputs.detach().cpu().numpy())

        # Concatenate predictions for this fold
        fold_preds = np.concatenate(fold_preds)

        # Add to ensemble accumulator
        final_preds += fold_preds

        # Cleanup to free GPU memory for the next model
        del model, state_dict
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Average Predictions
    avg_preds = final_preds / len(valid_model_paths)

    # 5. Post-processing
    # Clip to valid range [1, 6] and round to nearest integer
    final_scores = np.rint(np.clip(avg_preds, 1, 6)).astype(int)

    # 6. Generate Submission File
    # Load the test metadata to ensure correct essay_id mapping
    df_test = pd.read_csv(Config.test_path)

    submission = pd.DataFrame({"essay_id": df_test["essay_id"], "score": final_scores})

    # Define output path
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Save
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return submission_path
