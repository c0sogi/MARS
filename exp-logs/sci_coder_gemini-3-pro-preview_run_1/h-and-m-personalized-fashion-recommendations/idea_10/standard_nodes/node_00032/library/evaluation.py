import pandas as pd
import numpy as np
from library.utils import Timer


def calculate_map12(predictions, ground_truth, k=12):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12) for the given predictions and ground truth.

    This function implements the competition metric:
    MAP@12 = (1/U) * Sum_{u=1}^U (1/min(m, 12)) * Sum_{k=1}^min(n, 12) (P(k) * rel(k))

    Args:
        predictions (pd.DataFrame): DataFrame containing 'customer_id' and 'prediction' columns.
                                    'prediction' is a space-separated string of article IDs.
        ground_truth (pd.DataFrame): DataFrame containing 'customer_id' and 'article_id' columns.
                                     This can be the raw transaction history for the validation period
                                     (one row per purchase) or already grouped (one row per customer).
        k (int): The cutoff rank (default is 12).

    Returns:
        float: The MAP@12 score.
    """
    with Timer("MAP@12 Calculation"):
        # 1. Prepare Ground Truth
        # We need a list of actual article_ids per customer.
        if "article_id" in ground_truth.columns:
            # Check if the column already contains lists/arrays (grouped) or scalars (raw)
            # We sample the first non-null value to check type
            sample_val = None
            if not ground_truth.empty:
                sample_val = ground_truth["article_id"].iloc[0]

            if isinstance(sample_val, (list, np.ndarray, tuple)):
                # Already grouped
                truth_grouped = ground_truth.copy()
                truth_grouped = truth_grouped.rename(columns={"article_id": "actual"})
            else:
                # Raw transactions: Group by customer_id
                truth_grouped = (
                    ground_truth.groupby("customer_id")["article_id"]
                    .apply(list)
                    .reset_index()
                )
                truth_grouped = truth_grouped.rename(columns={"article_id": "actual"})
        else:
            raise ValueError("ground_truth DataFrame must contain 'article_id' column.")

        # 2. Prepare Predictions
        if "prediction" not in predictions.columns:
            raise ValueError("predictions DataFrame must contain 'prediction' column.")

        pred_df = predictions[["customer_id", "prediction"]].copy()
        # Fill NaN predictions with empty string
        pred_df["prediction"] = pred_df["prediction"].fillna("")
        # Convert space-separated string to list of strings
        pred_df["predicted"] = pred_df["prediction"].str.split()

        # 3. Merge Data
        # We calculate MAP over the set of users present in the ground truth (U)
        # Left join ensures we have rows for all validation customers
        merged = pd.merge(truth_grouped, pred_df, on="customer_id", how="left")

        # 4. Compute Average Precision (AP) per User
        actuals = merged["actual"].values
        predicteds = merged["predicted"].values

        scores = []

        # Iterate through users (zip is efficient enough for O(N) where N is number of users)
        for actual, predicted in zip(actuals, predicteds):
            # Handle case where user exists in truth but has no prediction (NaN -> None/float after merge)
            if not isinstance(predicted, list):
                predicted = []

            # Truncate predictions to k
            predicted = predicted[:k]

            # If no ground truth items, score is 0.0 (though usually validation sets filter these out)
            if not actual:
                scores.append(0.0)
                continue

            # Convert actuals to a set of strings for O(1) lookup and type consistency
            # This handles both int and string inputs robustly
            actual_set = set(str(x) for x in actual)

            score = 0.0
            num_hits = 0.0

            for i, p in enumerate(predicted):
                # Check if predicted item is in ground truth
                if str(p) in actual_set:
                    num_hits += 1.0
                    # Precision at rank i+1
                    score += num_hits / (i + 1.0)

            # Normalize by min(number of relevant items, k)
            denom = min(len(actual_set), k)

            if denom > 0:
                scores.append(score / denom)
            else:
                scores.append(0.0)

        # 5. Compute Mean
        final_score = np.mean(scores)

        # Print full precision as requested
        print(f"Validation MAP@12: {final_score}")

        return final_score
