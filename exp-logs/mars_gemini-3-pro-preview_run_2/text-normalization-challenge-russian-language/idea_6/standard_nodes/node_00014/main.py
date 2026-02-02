import pandas as pd
import torch
import os
import sys

# Import library modules
from library.config import Config
from library.cascade_manager import CascadeManager
from library.hfbb_engine import HFBB
from library.neural_net import (
    CharToSubwordTransformer,
    CharTokenizer,
    TargetBPETokenizer,
)
from library.training_engine import Trainer
from library.utils import set_seed


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Enable DEBUG to speed up Neural Network training data generation (Jackknifing)
    Config.DEBUG = True
    # Set a reasonable subset size (500k) to get enough residuals for training
    # while keeping execution time short (< 2 hours).
    Config.DEBUG_SIZE = 500000
    # Reduce epochs for speed
    Config.NUM_EPOCHS = 5

    print(
        f"Configuration Set: DEBUG={Config.DEBUG}, DEBUG_SIZE={Config.DEBUG_SIZE}, EPOCHS={Config.NUM_EPOCHS}"
    )

    # Initialize Manager
    manager = CascadeManager()

    # ==========================================
    # 2. Model Training
    # ==========================================

    # Train Tier 1: HFBB (Memory Model)
    # We train this on the FULL dataset because it's fast and provides the best baseline.
    # HFBB.fit() does not check Config.DEBUG, which is what we want here.
    manager.train_hfbb()

    # Train Tier 2: Residual Transformer (Neural Network)
    # We use the Trainer class directly to ensure clarity.
    # This will trigger residual generation. Since Config.DEBUG=True,
    # it will generate residuals from the first 500k rows of training data.
    print("Training Tier 2: Residual Transformer...")
    trainer = Trainer(debug=True)
    trainer.fit()

    # ==========================================
    # 3. Validation on Hold-out Set
    # ==========================================
    print("\nStarting Validation on Full Hold-out Set...")

    # Load full validation data (ignoring DEBUG flag for final evaluation)
    df_val = pd.read_csv(Config.VAL_FILE)

    # Load Trained HFBB Model
    hfbb = HFBB()
    hfbb.fit(load_cached_data=True)

    # Load Trained Neural Model
    device = torch.device(Config.DEVICE)
    if not os.path.exists(Config.TRANSFORMER_CHECKPOINT):
        print(
            "Warning: Transformer checkpoint not found. Validation will rely on HFBB only."
        )
        neural_model = None
        # Dummy tokenizers to prevent crash if model missing
        char_tokenizer = CharTokenizer()
        bpe_tokenizer = TargetBPETokenizer()
    else:
        checkpoint = torch.load(Config.TRANSFORMER_CHECKPOINT, map_location=device)

        # Reconstruct Tokenizers
        char_tokenizer = CharTokenizer()
        char_tokenizer.char2idx = checkpoint["char_vocab"]
        char_tokenizer.idx2char = {v: k for k, v in char_tokenizer.char2idx.items()}
        char_tokenizer.vocab_size = len(char_tokenizer.char2idx)

        bpe_tokenizer = TargetBPETokenizer()
        bpe_tokenizer.load()

        # Reconstruct Neural Model
        model_config = checkpoint["config"]
        neural_model = CharToSubwordTransformer(
            src_vocab_size=model_config["src_vocab_size"],
            tgt_vocab_size=model_config["tgt_vocab_size"],
            pad_idx=char_tokenizer.pad_token_id,
        ).to(device)
        neural_model.load_state_dict(checkpoint["model_state_dict"])
        neural_model.eval()

    # Run Inference using the Cascade Manager
    # This automatically routes: HFBB -> (if semiotic miss) Neural -> Identity
    val_preds = manager.run_inference(
        df_val, hfbb, neural_model, char_tokenizer, bpe_tokenizer
    )

    # Calculate Metric
    # Ensure strict string comparison
    targets = df_val["after"].fillna("").astype(str)
    predictions = val_preds.fillna("").astype(str)

    correct = (predictions == targets).sum()
    total = len(targets)
    accuracy = correct / total

    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")
    df_val["pred"] = predictions
    df_val["is_error"] = (df_val["pred"] != df_val["after"]).astype(int)

    # Correlation with Input Length
    # Longer tokens might be harder to normalize (e.g. complex dates)
    df_val["input_len"] = df_val["before"].astype(str).str.len()
    corr_len = df_val["is_error"].corr(df_val["input_len"])
    print(f"Correlation (Error vs Input Length): {corr_len}")

    # Error Rate by Class
    if "class" in df_val.columns:
        print("\nError Rate by Class (Top 5):")
        class_errors = (
            df_val.groupby("class")["is_error"].mean().sort_values(ascending=False)
        )
        print(class_errors.head(5))

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.9784022349361615

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy ({accuracy}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        manager.predict_and_submit()
    else:
        print(
            f"\nValidation accuracy ({accuracy}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
