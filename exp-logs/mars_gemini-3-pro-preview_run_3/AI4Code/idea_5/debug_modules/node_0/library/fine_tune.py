import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses
from library.config import Config
from library.utils import set_seed
from library.dataset import ContrastiveDataset, generate_training_pairs


def fine_tune_models(
    load_cached_data=True, epochs=Config.EPOCHS, batch_size=Config.TRAIN_BATCH_SIZE
):
    """
    Orchestrates the fine-tuning of the dual-backbone models using contrastive learning.

    Args:
        load_cached_data (bool): Whether to load pre-generated training pairs from cache.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    print("Loading training metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    print("Generating training pairs for contrastive learning...")
    # Generate (Markdown, Code) pairs
    # The generate_training_pairs function handles caching and debug sampling
    pairs_df = generate_training_pairs(
        df_train, load_cached_data=load_cached_data, debug=Config.DEBUG
    )

    print(f"Total training pairs: {len(pairs_df)}")

    # Create the Dataset (agnostic to the model)
    train_dataset = ContrastiveDataset(pairs_df)

    # Iterate through the defined backbones
    for model_name in Config.MODEL_NAMES:
        save_path = Config.MODEL_SAVE_PATHS[model_name]

        print(f"\n{'='*40}")
        print(f"Fine-tuning backbone: {model_name}")
        print(f"Output directory: {save_path}")
        print(f"{'='*40}")

        # Ensure output directory exists
        os.makedirs(save_path, exist_ok=True)

        # 1. Define Model Architecture
        # We explicitly define the Transformer and Pooling modules.
        # This is necessary for models like CodeBERT which are not default SentenceTransformers.
        word_embedding_model = models.Transformer(
            model_name, max_seq_length=Config.MAX_LEN
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=str(Config.DEVICE)
        )

        # 2. Create DataLoader with Model-Specific Collation
        # We create the DataLoader inside the loop to use model.smart_batching_collate
        # This ensures inputs are tokenized correctly for the specific model (e.g., CodeBERT vs MPNet)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=model.smart_batching_collate,
        )

        # 3. Define Loss Function
        # MultipleNegativesRankingLoss maximizes similarity for positive pairs
        # and minimizes it for other pairs in the batch (in-batch negatives).
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # 4. Train
        warmup_steps = int(len(train_dataloader) * epochs * Config.WARMUP_RATIO)

        print(f"Starting training for {epochs} epoch(s)...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": Config.LEARNING_RATE},
            weight_decay=Config.WEIGHT_DECAY,
            output_path=save_path,
            show_progress_bar=False,  # Suppress progress bars as required
        )

        print(f"Successfully saved fine-tuned {model_name} to {save_path}")
