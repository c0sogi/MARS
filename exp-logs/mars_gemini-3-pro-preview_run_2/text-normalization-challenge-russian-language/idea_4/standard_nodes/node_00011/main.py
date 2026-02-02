import pandas as pd
import numpy as np
import torch
import sys
import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, is_semiotic
from library.hfbb_engine import HFBBModel
from library.transformer_data import NormalizationDataset
from library.trainer import fit_transformer
from library.inference import HybridSystem


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for fast baseline execution
    Config.DEBUG = True
    Config.DEBUG_SIZE = 200000  # Train on a subset to meet time limits
    Config.NUM_EPOCHS = 2  # Limit epochs
    Config.BATCH_SIZE = 256  # Efficient batch size for A100

    set_seed(Config.SEED)

    print("Configuration set for fast baseline.")

    # ==========================================
    # 2. Train / Prepare Models
    # ==========================================
    # Explicitly train the transformer first with debug=True to ensure speed.
    # This creates the checkpoint that HybridSystem will load.
    print("Training Transformer (Tier 2) on subset...")
    fit_transformer(load_cached_data=True, debug=Config.DEBUG)

    # Initialize the Hybrid System
    print("Initializing Hybrid System...")
    system = HybridSystem()
    system.prepare_models()  # This loads HFBB and the Transformer we just trained

    # ==========================================
    # 3. Validation
    # ==========================================
    print("Loading Validation Data...")
    df_val = pd.read_csv(Config.VAL_DATA)

    # Generate context for validation data (using helper from system)
    # We need to temporarily treat it like test data to generate context columns
    df_val = system.generate_test_context(df_val)

    print("Running Validation Inference...")
    predictions = [None] * len(df_val)
    transformer_indices = []

    tokens = df_val["before"].values
    prevs = df_val["prev"].values
    nexts = df_val["next"].values

    # Pass 1: HFBB (Tier 1) and Gating
    for i in range(len(df_val)):
        token = str(tokens[i])
        prev_tok = str(prevs[i])
        next_tok = str(nexts[i])

        # Tier 1: HFBB
        norm = system.hfbb.get_normalization(token, prev_tok, next_tok)
        if norm is not None:
            predictions[i] = norm
            continue

        # Gate
        if is_semiotic(token):
            transformer_indices.append(i)
        else:
            # Tier 3: Identity Fallback
            predictions[i] = token

    # Pass 2: Transformer (Tier 2)
    if transformer_indices:
        print(f"Running Transformer on {len(transformer_indices)} validation tokens...")
        df_subset = df_val.iloc[transformer_indices].copy()

        # Create Dataset
        dataset = NormalizationDataset(
            df_subset, system.tokenizer, max_len=Config.MAX_SEQ_LEN, is_test=True
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        generated_texts = []
        with torch.no_grad():
            for src in loader:
                out_seqs = system.batch_greedy_decode(src, max_len=Config.MAX_SEQ_LEN)
                out_seqs = out_seqs.cpu().tolist()
                for seq in out_seqs:
                    text = system.tokenizer.decode(seq, skip_special_tokens=True)
                    generated_texts.append(text)

        # Assign predictions
        for idx, text in zip(transformer_indices, generated_texts):
            predictions[idx] = text

    # Fill any remaining Nones (safety)
    for i in range(len(predictions)):
        if predictions[i] is None:
            predictions[i] = str(tokens[i])

    # Calculate Metric
    actuals = df_val["after"].astype(str).values
    preds = np.array(predictions, dtype=str)

    # Exact string match accuracy
    accuracy = (preds == actuals).mean()
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("Performing Failure Analysis...")
    df_val["error"] = (preds != actuals).astype(int)
    df_val["len_before"] = df_val["before"].astype(str).apply(len)

    # Encode class for correlation
    le = LabelEncoder()
    df_val["class_enc"] = le.fit_transform(df_val["class"].astype(str))

    corr_len = df_val["len_before"].corr(df_val["error"])
    corr_class = df_val["class_enc"].corr(df_val["error"])

    print(f"Correlation Error vs Length: {corr_len}")
    print(f"Correlation Error vs Class: {corr_class}")

    # ==========================================
    # 5. Submission
    # ==========================================
    threshold = 0.9784022349361615
    if accuracy > threshold:
        print(
            f"Validation accuracy ({accuracy}) > threshold ({threshold}). Generating submission..."
        )
        system.generate_submission()
    else:
        print(
            f"Validation accuracy ({accuracy}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
