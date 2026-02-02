import os
import torch
import numpy as np
import random
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.data_utils import Tokenizer, build_embedding_matrix
from library.dataset import NQDataset
from library.model import GlobalContextPointwiseNet
from library.trainer import Trainer
from library.inference import Evaluator


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demonstration():
    print("=== Starting Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")
    # Override Config values to run on a tiny subset
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Only process 50 samples
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.VOCAB_SIZE = 1000  # Small vocab for demo
    Config.EMBEDDING_DIM = 16  # Small embedding dim
    Config.HIDDEN_DIM = 32

    # Ensure working directory is clean for this run to demonstrate creation
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, Epochs=1, Batch=4")

    # ---------------------------------------------------------
    # 2. Tokenizer and Vocabulary Generation
    # ---------------------------------------------------------
    print("\n[2] Demonstrating Tokenizer...")

    # Create some dummy text to fit the tokenizer
    dummy_texts = [
        "what is the capital of france",
        "the capital of france is paris",
        "who wrote harry potter",
        "jk rowling wrote harry potter",
        "is the sky blue",
        "yes the sky is blue",
    ]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dummy_texts, min_freq=1, vocab_size=Config.VOCAB_SIZE)

    # Verify tokenizer functionality
    seqs = tokenizer.texts_to_sequences(["france paris"])
    print(f"Tokenized 'france paris': {seqs}")
    assert len(seqs[0]) == 2, "Tokenizer failed to produce correct sequence length"

    # Save tokenizer to cache (required for Dataset/Trainer classes)
    tokenizer.save(Config.VOCAB_CACHE_FILE)
    assert os.path.exists(Config.VOCAB_CACHE_FILE), "Vocabulary cache file not created"
    print("Tokenizer vocabulary saved successfully.")

    # ---------------------------------------------------------
    # 3. Embedding Matrix Construction
    # ---------------------------------------------------------
    print("\n[3] Building Embedding Matrix...")
    embedding_matrix = build_embedding_matrix(
        tokenizer.word_index, embedding_dim=Config.EMBEDDING_DIM, load_cached_data=False
    )

    expected_shape = (tokenizer.vocab_size, Config.EMBEDDING_DIM)
    assert (
        embedding_matrix.shape == expected_shape
    ), f"Embedding matrix shape mismatch. Got {embedding_matrix.shape}, expected {expected_shape}"
    print(f"Embedding matrix built with shape: {embedding_matrix.shape}")

    # ---------------------------------------------------------
    # 4. Dataset Loading (Train Split)
    # ---------------------------------------------------------
    print("\n[4] Instantiating NQDataset (Train)...")
    # We use the provided metadata paths. The class handles loading.
    # load_cached_data=False forces processing of the raw JSONL (limited by DEBUG_SAMPLE_SIZE)
    train_dataset = NQDataset(
        metadata_path=Config.TRAIN_META_PATH,
        raw_data_path=Config.TRAIN_DATA_PATH,
        tokenizer=tokenizer,
        is_train=True,
        load_cached_data=False,  # Force processing
        debug=True,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    print(f"Dataset size: {len(train_dataset)}")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("Sample keys:", sample.keys())

        # Verify tensor shapes
        assert (
            sample["q_seq"].shape[0] == Config.MAX_Q_LEN
        ), "Question sequence padding incorrect"
        assert (
            sample["c_seq"].shape[0] == Config.MAX_DOC_LEN
        ), "Candidate sequence padding incorrect"
        assert isinstance(sample["long_label"], torch.Tensor), "Label is not a tensor"
        print("Dataset verification successful.")
    else:
        print("Warning: Dataset is empty (possibly due to filtering in debug mode).")

    # ---------------------------------------------------------
    # 5. Model Instantiation and Forward Pass
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Model Forward Pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GlobalContextPointwiseNet(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=0.1,
        embedding_matrix=embedding_matrix,
    ).to(device)

    # Create dummy batch
    batch_size = 2
    dummy_q = torch.randint(0, tokenizer.vocab_size, (batch_size, Config.MAX_Q_LEN)).to(
        device
    )
    dummy_c = torch.randint(
        0, tokenizer.vocab_size, (batch_size, Config.MAX_DOC_LEN)
    ).to(device)

    # Forward pass
    l_logits, s_logits, e_logits, yn_logits = model(dummy_q, dummy_c)

    # Verify output shapes
    assert l_logits.shape == (
        batch_size,
    ), f"Long logits shape mismatch: {l_logits.shape}"
    assert s_logits.shape == (
        batch_size,
        Config.MAX_DOC_LEN,
    ), f"Start logits shape mismatch: {s_logits.shape}"
    assert e_logits.shape == (
        batch_size,
        Config.MAX_DOC_LEN,
    ), f"End logits shape mismatch: {e_logits.shape}"
    assert yn_logits.shape == (
        batch_size,
        Config.NUM_YES_NO_CLASSES,
    ), f"Yes/No logits shape mismatch: {yn_logits.shape}"
    print("Model forward pass successful. Output shapes verified.")

    # ---------------------------------------------------------
    # 6. Trainer Execution
    # ---------------------------------------------------------
    print("\n[6] Running Trainer (Training Loop)...")
    # Initialize Trainer
    # It will reload the vocab we saved earlier and rebuild embeddings if needed
    trainer = Trainer(load_cached_data=True)

    # Run fit. This triggers the training loop on the datasets.
    # Since we set epochs=1 and debug=True, this should be fast.
    trainer.fit(load_cached_data=True)

    # Check if best model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Training completed. Best model saved at {best_model_path}")
    else:
        # If validation loss didn't improve (unlikely in 1 epoch starting from inf), it might not save.
        # But logic says if avg_val_loss < inf, it saves.
        print(
            "Training completed (Check: Model file might not exist if validation failed completely)."
        )

    # ---------------------------------------------------------
    # 7. Inference / Evaluator Execution
    # ---------------------------------------------------------
    print("\n[7] Running Evaluator (Inference)...")
    # Ensure we have a model file to load. If trainer didn't save one (e.g. empty dataset),
    # we save the current dummy model to allow evaluator to proceed.
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)
        print("Created temporary model checkpoint for inference demonstration.")

    evaluator = Evaluator(load_cached_data=True)

    # Run generation
    # This processes the test set (limited by DEBUG=True logic inside NQDataset if applicable,
    # though NQDataset debug flag is set in __init__. Evaluator inits dataset with defaults.
    # We need to ensure Evaluator uses the debug config we set globally.)
    evaluator.generate_submission()

    submission_path = Config.SUBMISSION_FILE
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission generated successfully at {submission_path}")
        print(f"Submission shape: {df_sub.shape}")
        print("First 5 rows:")
        print(df_sub.head())
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
