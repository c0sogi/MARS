import os
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processor import get_dataloader, get_tokenizer, SOS_TOKEN, EOS_TOKEN
from library.neural_agent import NeuralTrainer
from library.symbolic_agent import NgramMemory


def train_symbolic(config: Config):
    """
    Builds the symbolic N-gram memory.
    Leverages the caching mechanism in NgramMemory to avoid re-computation.
    """
    print("--- Symbolic Training ---")
    memory = NgramMemory(config)
    memory.build_stats(load_cached_data=True)
    return memory


def train_neural(config: Config):
    """
    Manages the training lifecycle of the neural character-level transformer.
    Includes data loading, tokenizer preparation, and the training loop.
    Implements a check to skip training if a valid checkpoint already exists.
    """
    print("--- Neural Training ---")

    # Check for existing checkpoint to save time during iterative development
    if os.path.exists(config.model_checkpoint_path):
        print(f"Checkpoint found at {config.model_checkpoint_path}. Skipping training.")
        return

    # Load DataLoaders
    # shuffle=True for training to ensure good stochastic gradient descent
    print("Loading DataLoaders...")
    train_loader = get_dataloader(config, split="train", load_cached=True, shuffle=True)
    val_loader = get_dataloader(config, split="val", load_cached=True, shuffle=False)

    # Load or Fit Tokenizer
    tokenizer = get_tokenizer(config)

    # Initialize Trainer
    trainer = NeuralTrainer(config, tokenizer)

    # Execute Training Loop (includes validation and early stopping)
    trainer.fit(train_loader, val_loader)

    return trainer


def generate_neural_predictions(config: Config) -> dict:
    """
    Generates predictions for the subset of test tokens identified as requiring neural normalization
    (i.e., tokens containing digits).

    Returns:
        dict: A mapping from 'sentence_id_token_id' to 'normalized_text'.
    """
    print("--- Neural Inference ---")

    # Load Tokenizer & Model
    tokenizer = get_tokenizer(config)
    trainer = NeuralTrainer(config, tokenizer)

    try:
        trainer.load_model(config.model_checkpoint_path)
    except FileNotFoundError:
        print(
            "Warning: No neural model checkpoint found. Neural predictions will be empty."
        )
        return {}

    # Load Test Data (Subset of digit-containing tokens)
    # shuffle=False is strictly required to maintain ID mapping if needed,
    # though we use explicit IDs from the batch.
    test_loader = get_dataloader(config, split="test", load_cached=True, shuffle=False)

    preds = {}
    device = get_device()
    trainer.model.eval()

    print(f"Generating predictions for {len(test_loader.dataset)} sequences...")

    with torch.no_grad():
        for batch in test_loader:
            batch_ids = batch["id"]
            src = batch["src"].to(device)

            # Generate sequences
            # Output shape: [batch_size, generated_len]
            generated_ids = trainer.model.generate(
                src, max_len=config.max_seq_len, device=device
            )

            # Decode sequences
            for i, seq_ids in enumerate(generated_ids):
                seq_list = seq_ids.tolist()

                # Truncate at EOS token if present to remove padding/garbage
                if tokenizer.eos_token_id in seq_list:
                    eos_idx = seq_list.index(tokenizer.eos_token_id)
                    seq_list = seq_list[:eos_idx]

                # Decode to string (removes SOS and other special tokens)
                text = tokenizer.decode(seq_list, remove_special_tokens=True)

                preds[batch_ids[i]] = text

    return preds


def run_inference(config: Config):
    """
    Orchestrates the Hybrid Inference logic and generates the submission file.
    Implements the Priority Cascade: Trigram -> Neural -> Bigram -> Unigram -> Identity.
    """
    print("--- Hybrid Inference Pipeline ---")

    # 1. Load Symbolic Memory (Fast Lookup)
    memory = NgramMemory(config)
    memory.build_stats(load_cached_data=True)

    # 2. Generate Neural Predictions (Generalization for numbers/dates)
    neural_preds = generate_neural_predictions(config)

    # 3. Process Full Test Set
    print("Processing full test set for final submission...")
    test_path = os.path.join(config.metadata_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test metadata not found at {test_path}")

    df_test = pd.read_csv(test_path)
    # Ensure sorting to correctly reconstruct sentence context
    df_test = df_test.sort_values(["sentence_id", "token_id"])

    # Group by sentence to enable context extraction
    grouped = df_test.groupby("sentence_id")

    results = []

    # Iterate over sentences
    for sid, group in grouped:
        tokens = group["before"].fillna("").astype(str).tolist()
        token_ids = group["token_id"].tolist()
        seq_len = len(tokens)

        for i in range(seq_len):
            curr_tok = tokens[i]
            t_id = token_ids[i]
            full_id = f"{sid}_{t_id}"

            # Define Context
            prev_tok = tokens[i - 1] if i > 0 else SOS_TOKEN
            next_tok = tokens[i + 1] if i < seq_len - 1 else EOS_TOKEN

            # --- Priority Cascade ---

            # Step 1: Trigram (Specific Memory)
            # High precision for memorized phrases
            norm = memory.query_trigram(prev_tok, curr_tok, next_tok)

            if norm is None:
                # Step 2: Neural (Generalization)
                # Check if we have a neural prediction (implies token contained digits)
                if full_id in neural_preds:
                    norm = neural_preds[full_id]
                else:
                    # Step 3: Bigram (General Memory)
                    norm = memory.query_bigram(prev_tok, curr_tok)

                    if norm is None:
                        # Step 4: Unigram (Fallback Memory)
                        norm = memory.query_unigram(curr_tok)

                        if norm is None:
                            # Step 5: Identity (Fallback)
                            norm = curr_tok

            results.append({"id": full_id, "after": norm})

    # 4. Save Submission
    df_res = pd.DataFrame(results)
    print(f"Saving submission to {config.submission_path}...")
    df_res.to_csv(config.submission_path, index=False)
    print("Done.")


def run_pipeline(config: Config):
    """
    Main entry point to run the full training and inference pipeline.
    """
    seed_everything(config.seed)

    # Train components
    train_symbolic(config)
    train_neural(config)

    # Generate submission
    run_inference(config)
