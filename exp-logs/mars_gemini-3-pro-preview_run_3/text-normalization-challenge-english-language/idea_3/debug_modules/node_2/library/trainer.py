import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything
from library.data_manager import (
    load_parquet_data,
    prepare_neural_dataframe,
    get_tokenizer,
    NormalizationDataset,
    collate_fn,
)
from library.neural_solver import TransformerSeq2Seq, NeuralTrainer


def train_neural_model(load_cached_data=True):
    """
    Orchestrates the training of the neural Transformer model.

    1. Loads raw training and validation data.
    2. Preprocesses data (filtering 'easy' cases, adding context).
    3. Builds/Loads the character tokenizer.
    4. Initializes the TransformerSeq2Seq model.
    5. Runs the training loop with validation and early stopping.

    Args:
        load_cached_data (bool): If True, attempts to load processed data/stats from cache.
                                 If False, re-computes everything.

    Returns:
        model (nn.Module): The trained model (loaded with best weights).
        tokenizer (CharTokenizer): The tokenizer used.
    """

    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting neural model training pipeline on {device}...")

    # 2. Load Raw Metadata
    # We need the raw data first to establish the tokenizer vocabulary
    # and then to filter for the specific neural dataset.
    print("Loading raw metadata...")
    df_train_raw = load_parquet_data(split="train")
    df_val_raw = load_parquet_data(split="val")

    # 3. Tokenizer Management
    # We fit the tokenizer on the full raw training set to ensure the vocabulary
    # handles all characters seen in the corpus, even if some 'easy' tokens
    # are filtered out of the specific neural training set.
    tokenizer = get_tokenizer(df_train=df_train_raw, load_cached_data=load_cached_data)
    print(f"Tokenizer ready. Vocab size: {len(tokenizer)}")

    # 4. Prepare Neural Datasets
    # This step filters the raw data to keep only "hard" cases (non-plain/punct or complex chars)
    # and formats the context window columns.
    df_train_neural = prepare_neural_dataframe(
        df_train_raw, split="train", load_cached_data=load_cached_data
    )
    df_val_neural = prepare_neural_dataframe(
        df_val_raw, split="val", load_cached_data=load_cached_data
    )

    # 5. Create PyTorch Datasets and Loaders
    print("Creating Datasets and DataLoaders...")

    # Mode="train" ensures we get (source, target) tuples
    train_dataset = NormalizationDataset(
        df_train_neural, tokenizer, max_len=Config.MAX_CHAR_LEN, mode="train"
    )
    val_dataset = NormalizationDataset(
        df_val_neural, tokenizer, max_len=Config.MAX_CHAR_LEN, mode="train"
    )

    print(f"Neural Training Samples: {len(train_dataset)}")
    print(f"Neural Validation Samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # 6. Initialize Model
    print("Initializing TransformerSeq2Seq model...")
    model = TransformerSeq2Seq(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
    )

    # 7. Run Training
    trainer = NeuralTrainer(model, tokenizer, device=device)
    trainer.train(train_loader, val_loader, epochs=Config.EPOCHS)

    print("Training pipeline complete.")
    return trainer.model, tokenizer
