import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.metrics import accuracy_score

# Import provided library components
from library.config import Config
from library.utils import set_seed
from library.hfbb import HierarchicalBackoff
from library.tokenizer import HybridTokenizer
from library.transformer_model import TransformerTrainer
from library.predictor import HybridPredictor


def main():
    # ==========================================
    # 1. Configuration Overrides for Fast Baseline
    # ==========================================
    # Limit training data and epochs to meet time constraints while maintaining quality
    Config.DEBUG = True
    Config.DEBUG_SIZE = 300000  # Train on 300k semiotic samples (robust baseline)
    Config.NUM_EPOCHS = 3  # Sufficient convergence for this dataset size
    Config.BATCH_SIZE = 256  # Efficient training on A100

    # Re-setup Config to update artifact paths based on the new configuration hash
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(
        f"Running with Config: DEBUG={Config.DEBUG}, EPOCHS={Config.NUM_EPOCHS}, SIZE={Config.DEBUG_SIZE}"
    )

    # ==========================================
    # 2. Component Initialization & Training
    # ==========================================
    print("\n--- Initializing Components ---")

    # Tokenizer (Train on full data for complete vocab)
    tokenizer = HybridTokenizer()
    tokenizer.train(load_cached_data=True)

    # HFBB Tier 1 Memory (Fit on full data for max coverage)
    hfbb = HierarchicalBackoff()
    hfbb.fit(load_cached_data=True)

    # Transformer Tier 2 (Train on filtered 'Residual' data)
    print("\n--- Training Transformer ---")
    trainer = TransformerTrainer(tokenizer)
    trainer.train(hfbb_model=hfbb)

    # ==========================================
    # 3. Validation
    # ==========================================
    print("\n--- Performing Full Validation ---")

    # Optimize batch size for inference (no gradients = less memory)
    Config.BATCH_SIZE = 2048

    # Initialize Predictor (automatically loads the best model checkpoint)
    predictor = HybridPredictor(load_cached_data=True)

    # Load the full validation set (not subsampled)
    df_val = pd.read_csv(Config.VAL_DATA)

    # Generate predictions using the full Hybrid Cascade
    # (HFBB -> Confidence Check -> Transformer -> Identity)
    preds = predictor.predict(df_val)

    # Calculate Metric (Exact String Match)
    targets = df_val["after"].fillna("").astype(str).tolist()
    preds = [str(p) if p is not None else "" for p in preds]

    acc = accuracy_score(targets, preds)
    print(f"Final Validation Metric: {acc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")

    # Create analysis dataframe
    df_analysis = df_val.copy()
    df_analysis["pred"] = preds
    df_analysis["target"] = targets
    # Binary error: 1 if incorrect, 0 if correct
    df_analysis["error"] = (df_analysis["pred"] != df_analysis["target"]).astype(int)

    # Feature 1: Input Length
    df_analysis["len_before"] = df_analysis["before"].fillna("").astype(str).apply(len)

    # Feature 2: Class (Encoded)
    if "class" in df_analysis.columns:
        df_analysis["class_code"] = df_analysis["class"].astype("category").cat.codes
    else:
        df_analysis["class_code"] = 0

    # Calculate correlations
    corr_len = df_analysis["error"].corr(df_analysis["len_before"])
    corr_class = df_analysis["error"].corr(df_analysis["class_code"])

    print("Correlation of Error with Input Features:")
    print(f"  Length (Characters): {corr_len:.4f}")
    print(f"  Class (Encoded): {corr_class:.4f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.9887932804852236

    if acc > THRESHOLD:
        print(f"\nValidation accuracy {acc} > {THRESHOLD}. Generating submission...")

        # Load test data
        df_test = pd.read_csv(Config.TEST_DATA)

        # Predict on test set
        test_preds = predictor.predict(df_test)

        # Format submission
        # ID format: sentence_id_token_id
        ids = df_test.apply(lambda x: f"{x['sentence_id']}_{x['token_id']}", axis=1)
        sub_df = pd.DataFrame({"id": ids, "after": test_preds})

        # Save to submission directory
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nValidation accuracy {acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
