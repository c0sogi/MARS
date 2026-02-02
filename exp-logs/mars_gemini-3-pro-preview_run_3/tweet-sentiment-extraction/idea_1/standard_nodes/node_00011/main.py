import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr

# Import functions and classes from the provided library
from library.config import SEED, SUBMISSION_PATH
from library.utils import set_seed, jaccard
from library.data_loader import load_datasets
from library.model import SentimentRelevanceModel


def main():
    # 1. Setup and Reproducibility
    # Ensure consistent results across runs
    set_seed(SEED)

    # 2. Data Loading
    # Load datasets using the cached preprocessed data for efficiency
    print("Loading datasets...")
    train_df, val_df, test_df = load_datasets(load_cached_data=True)

    # 3. Model Training
    # Initialize the statistical sentiment relevance model
    # This model learns word probabilities for positive/negative classes
    print("Initializing and training model...")
    model = SentimentRelevanceModel()
    model.fit(train_df, load_cached_data=True)

    # 4. Validation
    # Generate predictions on the hold-out validation set
    print("Running validation inference...")
    val_preds_df = model.predict(val_df)

    # Calculate Jaccard scores for each validation sample
    val_scores = []
    for i in range(len(val_df)):
        target_text = val_df.iloc[i]["selected_text"]
        pred_text = val_preds_df.iloc[i]["selected_text"]
        score = jaccard(target_text, pred_text)
        val_scores.append(score)

    # Compute and print the final mean metric
    final_metric = np.mean(val_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (0.0 = perfect match, 1.0 = no overlap)
    errors = 1.0 - np.array(val_scores)

    # Extract input features for correlation analysis
    # Ensure text is treated as string
    val_text_series = val_df["text"].astype(str)

    # Feature 1: Character length
    char_lengths = val_text_series.apply(len).values

    # Feature 2: Word count (splitting by whitespace)
    word_counts = val_text_series.apply(lambda x: len(x.split())).values

    # Calculate Pearson correlation between Error Magnitude and Input Features
    corr_char, _ = pearsonr(errors, char_lengths)
    corr_word, _ = pearsonr(errors, word_counts)

    print("Correlation between Error Magnitude (1 - Jaccard) and Input Features:")
    print(f"  Text Length (Characters): {corr_char:.6f}")
    print(f"  Text Length (Words):      {corr_word:.6f}")

    # Analyze error distribution by sentiment class
    print("\nMean Error by Sentiment Class:")
    val_df_analysis = val_df.copy()
    val_df_analysis["error"] = errors
    print(val_df_analysis.groupby("sentiment")["error"].mean())

    # 6. Submission Generation
    # Generate predictions for the test set and save to CSV only if metric improves
    if final_metric > 0.6962227540030603:
        print("\nGenerating submission for test set...")
        model.generate_submission(test_df)

        # Verify the output
        if pd.read_csv(SUBMISSION_PATH).shape[0] == len(test_df):
            print(f"Submission successfully saved to {SUBMISSION_PATH}")
        else:
            print("Warning: Submission file row count mismatch.")
    else:
        print(
            f"\nValidation metric ({final_metric:.6f}) did not exceed threshold (0.696223). Skipping submission."
        )


if __name__ == "__main__":
    main()
