import pandas as pd
import numpy as np
import torch
import os
import gc
from library.config import Config
from library.feature_generator import RankerFeatureFactory
from library.ranker_model import LGBMRanker
from library.data_utils import load_dataset

# Set seeds
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


def sigmoid(x):
    """Applies sigmoid function to convert logits to probabilities."""
    return 1 / (1 + np.exp(-x))


def map_at_12(predictions_df, ground_truth_df):
    """
    Calculates Mean Average Precision @ 12.

    Args:
        predictions_df: DataFrame with ['customer_id', 'prediction']
        ground_truth_df: DataFrame with ['customer_id', 'article_id']
    """
    # Group ground truth into a dictionary of sets for O(1) lookup
    # Using set for ground truth items per user (assuming unique relevance)
    truth = ground_truth_df.groupby("customer_id")["article_id"].apply(set).to_dict()

    # Map predictions for fast lookup
    preds_map = predictions_df.set_index("customer_id")["prediction"].to_dict()

    score_sum = 0.0
    n_users = len(truth)

    for cust_id, actuals in truth.items():
        if not actuals:
            n_users -= 1
            continue

        pred_str = preds_map.get(cust_id, "")
        if not pred_str:
            preds = []
        else:
            # Convert string IDs back to int for comparison if needed,
            # or ensure truth is string. truth is int from load_dataset usually.
            # predictions are strings in the submission format, but here we might have ints
            # Let's handle the string splitting carefully.
            try:
                preds = [int(x) for x in pred_str.split()]
            except ValueError:
                preds = []

        # Truncate to top 12
        preds = preds[:12]

        hits = 0
        sum_precisions = 0
        m = len(actuals)

        for k, p in enumerate(preds):
            if p in actuals:
                hits += 1
                sum_precisions += hits / (k + 1)

        # MAP formula denominator is min(m, 12)
        ap = sum_precisions / min(m, 12)
        score_sum += ap

    return score_sum / n_users if n_users > 0 else 0.0


def run():
    print("Initializing pipeline...")
    Config.setup()

    # 1. Data Loading & Generation
    print("Generating/Loading datasets...")
    factory = RankerFeatureFactory()

    # Load Train and Validation sets
    # We use cached data if available to speed up execution
    train_df = factory.create_train_dataset(load_cached_data=True)
    val_df = factory.create_validation_dataset(load_cached_data=True)

    # Optimization: Downsample Training Data
    # LightGBM ranker can be slow with millions of rows.
    # We keep all positives (label=1) and sample negatives (label=0).
    TARGET_TRAIN_SIZE = 2000000
    if len(train_df) > TARGET_TRAIN_SIZE:
        print(
            f"Downsampling training data from {len(train_df)} to ~{TARGET_TRAIN_SIZE}..."
        )
        pos_mask = train_df["label"] == 1
        neg_mask = train_df["label"] == 0

        positives = train_df[pos_mask]
        negatives = train_df[neg_mask]

        # Calculate how many negatives to keep
        n_pos = len(positives)
        n_neg_keep = TARGET_TRAIN_SIZE - n_pos

        if n_neg_keep > 0:
            negatives = negatives.sample(n=n_neg_keep, random_state=Config.SEED)
            train_df = pd.concat([positives, negatives])
            # Shuffle
            train_df = train_df.sample(frac=1, random_state=Config.SEED).reset_index(
                drop=True
            )
            print(f"New training size: {len(train_df)}")
        else:
            print("Warning: Positives exceed target size. Using all positives only.")
            train_df = positives.sample(frac=1, random_state=Config.SEED).reset_index(
                drop=True
            )

    # 2. Model Training
    print("Training Ranker...")
    ranker = LGBMRanker()
    ranker.train(train_df, val_df)

    # Clean up train_df to free memory
    del train_df
    gc.collect()

    # 3. Validation Evaluation
    print("Evaluating on Validation Set...")

    # We need to generate predictions for the validation set manually to compute MAP@12
    # ranker.model is the trained Booster
    model_features = ranker.model.feature_name()

    # Ensure val_df has the features
    X_val = val_df[model_features]
    val_scores = ranker.model.predict(X_val)

    # Add scores to a copy
    val_preds_df = val_df[["customer_id", "article_id"]].copy()
    val_preds_df["score"] = val_scores

    # Sort by score descending
    val_preds_df = val_preds_df.sort_values(
        ["customer_id", "score"], ascending=[True, False]
    )

    # Take top 12
    top_12_val = val_preds_df.groupby("customer_id").head(12)

    # Format as space-separated string
    val_submission = (
        top_12_val.groupby("customer_id")["article_id"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    val_submission.columns = ["customer_id", "prediction"]

    # Load Ground Truth (Actual transactions from the validation period)
    _, val_truth_df, _, _, _ = load_dataset()

    # Compute MAP@12
    final_map = map_at_12(val_submission, val_truth_df)
    print(f"Final Validation Metric: {final_map:.10f}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Error: |Label - Prob|
    # Since LambdaRank outputs unbounded scores, we apply sigmoid to get pseudo-probability
    val_df_analysis = val_df.copy()
    val_df_analysis["score"] = val_scores
    val_df_analysis["prob"] = sigmoid(val_scores)
    val_df_analysis["error"] = np.abs(
        val_df_analysis["label"] - val_df_analysis["prob"]
    )

    # Correlation with features
    # Select only numeric columns
    numeric_cols = val_df_analysis.select_dtypes(include=[np.number]).columns
    # Exclude non-feature columns
    exclude_cols = ["label", "score", "prob", "error", "article_id"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = val_df_analysis[feature_cols].corrwith(val_df_analysis["error"])
    correlations = correlations.sort_values(key=abs, ascending=False)

    print("Correlation between Error Magnitude and Input Features (Top 10):")
    print(correlations.head(10))

    # 5. Submission
    THRESHOLD = 0.026059042
    if final_map > THRESHOLD:
        print(
            f"\nValidation metric ({final_map:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate Inference Dataset
        test_df = factory.create_inference_dataset(load_cached_data=True)

        # Predict and Save
        ranker.predict(test_df)
    else:
        print(
            f"\nValidation metric ({final_map:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
