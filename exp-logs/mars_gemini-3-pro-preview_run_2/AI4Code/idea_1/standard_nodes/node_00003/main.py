import os
import pandas as pd
import numpy as np
import torch
import time
from scipy.stats import pearsonr

# Import provided library modules
from library.config import set_seed, VAL_METADATA_PATH, SUBMISSION_PATH, RANDOM_STATE
from library.data_loader import get_regression_data, get_inference_data
from library.model import RankPredictor, predict_notebook_order
from library.metrics import score_dataset


def failure_analysis(model, df_val_reg):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between prediction error and text features.
    """
    print("\n--- Performing Failure Analysis ---")

    # Ensure we have text data
    texts = df_val_reg["text"].fillna("").astype(str).tolist()
    true_ranks = df_val_reg["rank"].values

    # Predict
    pred_ranks = model.predict(texts)

    # Calculate Error
    errors = np.abs(pred_ranks - true_ranks)

    # Extract Features
    char_lengths = np.array([len(t) for t in texts])
    word_counts = np.array([len(t.split()) for t in texts])

    # Calculate Correlations
    # Handle edge cases where variance might be 0
    if np.std(errors) == 0 or np.std(char_lengths) == 0:
        corr_char = 0.0
    else:
        corr_char, _ = pearsonr(errors, char_lengths)

    if np.std(errors) == 0 or np.std(word_counts) == 0:
        corr_word = 0.0
    else:
        corr_word, _ = pearsonr(errors, word_counts)

    print(f"Mean Absolute Error (MAE): {np.mean(errors):.6f}")
    print(f"Correlation (Error vs Char Length): {corr_char:.6f}")
    print(f"Correlation (Error vs Word Count):  {corr_word:.6f}")

    if abs(corr_char) > 0.1:
        print(
            "Insight: There is a noticeable correlation between cell length and error."
        )
    else:
        print("Insight: Error appears relatively independent of cell length.")
    print("-----------------------------------\n")


def main():
    # 1. Setup
    set_seed(RANDOM_STATE)

    # Detect Device (Requirement compliance, though Ridge is CPU-based)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Process running. Detected device: {device}")

    # 2. Training
    print("Step 1: Loading Training Data...")
    # Cite solution_lesson_node_00002: Maximize Data Scale (use full dataset)
    df_train = get_regression_data(
        data_type="train", load_cached_data=True, max_samples=None
    )

    print(f"Step 2: Training Model on {len(df_train)} samples...")
    start_time = time.time()
    model = RankPredictor()
    model.fit(df_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # 3. Validation
    print("Step 3: Running Validation Inference...")
    # We validate on all available validation notebooks to get a proper metric
    val_notebooks = get_inference_data(data_type="val", max_samples=None)

    val_preds = []
    for nb in val_notebooks:
        order_str = predict_notebook_order(model, nb)
        val_preds.append({"id": nb["id"], "cell_order": order_str})

    df_val_pred = pd.DataFrame(val_preds)

    # Load Ground Truth
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)
    # Filter metadata to match the notebooks we actually predicted (in case of subsetting)
    df_val_meta = df_val_meta[df_val_meta["id"].isin(df_val_pred["id"])]

    # Calculate Metric
    metric_score = score_dataset(df_val_meta, df_val_pred)
    print(f"Final Validation Metric: {metric_score}")

    # 4. Failure Analysis
    # Load validation data in regression format (cell-level)
    df_val_reg = get_regression_data(
        data_type="val", load_cached_data=True, max_samples=10000
    )
    failure_analysis(model, df_val_reg)

    # 5. Submission
    print("Step 5: Generating Submission...")

    if metric_score > 0.7379338924270245:
        test_notebooks = get_inference_data(data_type="test", max_samples=None)

        if len(test_notebooks) == 0:
            print(
                "Warning: No test notebooks found. Creating empty submission or using sample."
            )
            # Fallback to creating an empty file with header if no data found (unlikely in competition)
            pd.DataFrame(columns=["id", "cell_order"]).to_csv(
                SUBMISSION_PATH, index=False
            )
        else:
            test_preds = []
            for nb in test_notebooks:
                order_str = predict_notebook_order(model, nb)
                test_preds.append({"id": nb["id"], "cell_order": order_str})

            df_submission = pd.DataFrame(test_preds)

            # Ensure directory exists
            os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

            df_submission.to_csv(SUBMISSION_PATH, index=False)
            print(f"Submission saved to {SUBMISSION_PATH}")
            print(f"Generated predictions for {len(df_submission)} notebooks.")
    else:
        print(
            f"Metric score {metric_score} did not exceed threshold. Skipping submission."
        )


if __name__ == "__main__":
    main()
