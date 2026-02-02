import os
import shutil
import pandas as pd
import torch
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data_factory import DataFactory
from library.symbolic_layer import SymbolicMemory
from library.neural_net import MultiTaskSeq2Seq
from library.training_agent import Trainer
from library.inference_manager import CascadePredictor


def create_mini_datasets():
    """
    Creates smaller versions of the metadata parquet files to ensure
    the demonstration runs quickly.
    """
    print("Creating mini datasets for rapid demonstration...")

    # Define paths
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.parquet")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.parquet")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.parquet")

    # Load head of original files
    # We use a small number of rows (e.g., 5000)
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH).head(5000)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH).head(1000)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH).head(1000)

    # Save mini files
    df_train.to_parquet(mini_train_path, index=False)
    df_val.to_parquet(mini_val_path, index=False)
    df_test.to_parquet(mini_test_path, index=False)

    # Override Config paths to point to mini datasets
    Config.TRAIN_DATA_PATH = mini_train_path
    Config.VAL_DATA_PATH = mini_val_path
    Config.TEST_DATA_PATH = mini_test_path

    print(f"Mini datasets created at {Config.WORKING_DIR}")


def override_config_for_speed():
    """
    Overrides default configuration with lightweight hyperparameters.
    """
    print("Overriding configuration for speed...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 5000
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.HIDDEN_DIM = 32
    Config.EMBEDDING_DIM = 16
    Config.ENC_LAYERS = 1
    Config.DEC_LAYERS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Ensure directories exist
    Config.setup_environment()


def demo_data_factory():
    print("\n=== Demo: DataFactory ===")
    factory = DataFactory()

    # 1. Test Data Processing and Loader Generation
    # load_cached_data=False forces the factory to process the raw mini parquet file
    train_loader = factory.get_train_loader(load_cached_data=False)

    # 2. Verify Tokenizer and Encoder Fitting
    assert (
        factory.tokenizer_fitted
    ), "Tokenizer should be fitted after getting train loader."
    assert (
        factory.encoder_fitted
    ), "LabelEncoder should be fitted after getting train loader."
    assert (
        len(factory.tokenizer) > 5
    ), "Vocabulary should contain at least special tokens."

    # 3. Verify Batch Structure
    src, tgt, cls = next(iter(train_loader))
    print(
        f"Batch Shapes - Source: {src.shape}, Target: {tgt.shape}, Class: {cls.shape}"
    )

    assert src.shape[0] == Config.BATCH_SIZE, "Batch size mismatch."
    assert src.dim() == 2, "Source tensor should be 2D (batch, seq_len)."
    assert tgt.dim() == 2, "Target tensor should be 2D (batch, seq_len)."

    return factory


def demo_symbolic_memory(factory):
    print("\n=== Demo: SymbolicMemory ===")
    memory = SymbolicMemory()

    # Load the processed dataframe created by DataFactory
    # DataFactory saves to 'train_processed.parquet' in WORKING_DIR
    processed_train_path = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    df_processed = pd.read_parquet(processed_train_path)

    # 1. Build Statistics
    # We force build from the dataframe to verify logic
    memory.build_stats(df=df_processed, load_cached_data=False)

    # 2. Verify Stats Content
    print(f"Trigrams: {len(memory.trigram_stats)}")
    print(f"Unigrams: {len(memory.unigram_stats)}")

    assert len(memory.unigram_stats) > 0, "Unigram stats should not be empty."

    # 3. Test Query
    # Pick a sample from the dataframe to query
    sample = df_processed.iloc[0]
    prev, curr, next_tok = sample["prev"], sample["before"], sample["next"]
    target = sample["after"]

    prediction = memory.query(prev, curr, next_tok)
    print(
        f"Query: '{curr}' (Context: {prev}, {next_tok}) -> Prediction: '{prediction}' (Target: '{target}')"
    )

    # Note: Prediction might be None if stats didn't capture it (unlikely with same data),
    # or it might match. We just verify the method runs without error.


def demo_neural_network(factory):
    print("\n=== Demo: Neural Network ===")
    vocab_size = len(factory.tokenizer)

    # 1. Instantiate Model
    model = MultiTaskSeq2Seq(vocab_size).to(Config.DEVICE)

    # 2. Create Dummy Input
    batch_size = 4
    seq_len = 10
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(Config.DEVICE)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len)).to(Config.DEVICE)

    # 3. Forward Pass
    decoder_outputs, aux_outputs = model(src, tgt)

    print(f"Output Shapes - Decoder: {decoder_outputs.shape}, Aux: {aux_outputs.shape}")

    # 4. Assertions
    assert decoder_outputs.shape == (
        batch_size,
        seq_len,
        vocab_size,
    ), "Decoder output shape mismatch."
    assert aux_outputs.shape == (
        batch_size,
        Config.NUM_AUX_CLASSES,
    ), "Auxiliary output shape mismatch."


def demo_training_agent():
    print("\n=== Demo: Training Agent ===")

    # 1. Initialize Trainer
    # Uses the mini datasets configured earlier
    trainer = Trainer(debug=True, load_cached_data=True)

    # 2. Run Training Loop (1 Epoch)
    trainer.fit(epochs=1)

    # 3. Verify Model Saving
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("Training complete and model saved.")


def demo_inference_manager():
    print("\n=== Demo: Inference Manager ===")

    predictor = CascadePredictor()

    # 1. Validate on Mini Validation Set
    # load_cached_data=False ensures we process the mini_val.parquet
    accuracy = predictor.validate(load_cached_data=False)
    print(f"Validation Accuracy on Mini Set: {accuracy:.4f}")

    # 2. Generate Submission on Mini Test Set
    df_submission = predictor.generate_submission(load_cached_data=False)

    # 3. Verify Submission
    print(f"Submission Shape: {df_submission.shape}")
    print(df_submission.head())

    assert (
        "id" in df_submission.columns and "after" in df_submission.columns
    ), "Submission columns missing."
    assert (
        len(df_submission) == 1000
    ), "Submission should have 1000 rows (matching mini_test)."
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not saved."


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Setup environment for fast demo
    override_config_for_speed()
    create_mini_datasets()

    # Run Demos
    try:
        factory = demo_data_factory()
        demo_symbolic_memory(factory)
        demo_neural_network(factory)
        demo_training_agent()
        demo_inference_manager()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        raise e
