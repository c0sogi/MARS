import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything
from library.dataset import load_and_prepare_data, PearsonDataset
from library.model import CustomModel
from library.engine import valid_fn


def predict(debug: bool = False):
    """
    Performs inference on the test set using the trained models from all folds.
    Generates the submission file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    seed_everything(Config.seed)

    print(f"\n{'='*20} Inference on Test Set {'='*20}")

    # 1. Load and Prepare Data
    # load_and_prepare_data handles the caching and CPC context mapping
    df_test = load_and_prepare_data(Config.test_path)

    if debug:
        print("Debug mode: Sampling subset of test data.")
        df_test = df_test.iloc[:100].reset_index(drop=True)

    print(f"Total test samples: {len(df_test)}")

    # 2. Setup Tokenizer and DataLoader
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    test_dataset = PearsonDataset(df_test, tokenizer, max_length=Config.max_length)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Ensemble Inference
    # Initialize array to store accumulated predictions
    final_predictions = np.zeros(len(df_test))

    # Check how many folds were actually trained/saved
    # If a specific fold model is missing, we skip it (though ideally all should exist)
    folds_found = 0

    for fold in range(Config.n_fold):
        model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Predicting with model fold {fold}...")

        # Initialize model structure without pretrained weights (loading state_dict instead)
        model = CustomModel(pretrained=False)

        # Load weights
        state_dict = torch.load(model_path, map_location=Config.device)
        model.load_state_dict(state_dict)

        model.to(Config.device)

        # Run inference
        preds, _ = valid_fn(test_loader, model, Config.device)

        # Accumulate predictions
        # We will divide by the total count of valid models later
        final_predictions += preds
        folds_found += 1

        # Cleanup to free GPU memory
        del model, state_dict
        torch.cuda.empty_cache()
        gc.collect()

    if folds_found > 0:
        final_predictions /= folds_found
    else:
        print("Error: No models found for inference.")
        return

    # 4. Generate Submission File
    submission = pd.DataFrame({"id": df_test["id"], "score": final_predictions})

    # Clip scores to valid range [0, 1] as per metric definition
    submission["score"] = submission["score"].clip(0.0, 1.0)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print(submission.head())
