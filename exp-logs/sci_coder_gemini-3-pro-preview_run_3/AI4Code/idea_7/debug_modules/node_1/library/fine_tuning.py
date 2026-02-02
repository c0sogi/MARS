import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

from library.config import Config
from library.data_utils import get_data_splits, generate_relaxed_pairs, set_seed


def run_fine_tuning():
    """
    Executes the fine-tuning pipeline for the semantic backbone.
    1. Loads training metadata.
    2. Generates (Markdown, Code) pairs using the Relaxed Proximity strategy.
    3. Splits pairs into training and validation sets.
    4. Fine-tunes the MPNet model using Multiple Negatives Ranking Loss.
    5. Saves the best model to disk.
    """
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print("Initializing Fine-Tuning Phase...")

    # 1. Load Data Splits
    # We only use the training split for fine-tuning the backbone.
    # The validation split from metadata is reserved for the downstream ranking task evaluation.
    df_train, _, _ = get_data_splits()
    print(f"Loaded {len(df_train)} training notebooks.")

    # 2. Generate Training Pairs
    # We force regeneration (load_cached_data=False) to ensure the pairs correspond
    # exactly to the current df_train (which might be a debug subset).
    print("Generating relaxed proximity pairs...")
    pairs_df = generate_relaxed_pairs(df_train, load_cached_data=False)

    if pairs_df is None or len(pairs_df) == 0:
        print("Error: No pairs generated. Aborting fine-tuning.")
        return

    print(f"Generated {len(pairs_df)} pairs.")

    # 3. Create Train/Validation Split for Model Training
    # We split the generated pairs to monitor the contrastive loss and prevent overfitting.
    # Using a 90/10 split.
    train_df, val_df = train_test_split(
        pairs_df, test_size=0.1, random_state=Config.SEED, shuffle=True
    )
    print(
        f"Split into {len(train_df)} training pairs and {len(val_df)} validation pairs."
    )

    # 4. Prepare Training Data
    train_examples = []
    for _, row in train_df.iterrows():
        train_examples.append(
            InputExample(texts=[str(row["markdown"]), str(row["code"])])
        )

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=Config.TRAIN_BATCH_SIZE
    )

    # 5. Prepare Validation Data (Evaluator)
    # We use BinaryClassificationEvaluator.
    # Positives: The actual pairs from val_df (Label 1)
    # Negatives: Markdown from val_df paired with random Code from val_df (Label 0)
    val_examples = []
    val_md = val_df["markdown"].tolist()
    val_code = val_df["code"].tolist()

    # Create positives
    for m, c in zip(val_md, val_code):
        val_examples.append(InputExample(texts=[str(m), str(c)], label=1.0))

    # Create negatives by shuffling code
    # We use a local Random instance to avoid affecting global state if not necessary,
    # but set_seed handled global.
    val_code_shuffled = val_code.copy()
    random.shuffle(val_code_shuffled)

    for m, c in zip(val_md, val_code_shuffled):
        val_examples.append(InputExample(texts=[str(m), str(c)], label=0.0))

    # Initialize Evaluator
    # We use the BinaryClassificationEvaluator as it aligns well with the goal of
    # distinguishing correct code matches from incorrect ones.
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        val_examples, name="mpnet-fine-tune-eval"
    )

    # 6. Initialize Model
    print(f"Loading backbone model: {Config.MODEL_CHECKPOINT}")
    model = SentenceTransformer(Config.MODEL_CHECKPOINT)

    # 7. Define Loss
    # MultipleNegativesRankingLoss is ideal for (query, positive) pairs where
    # in-batch negatives are used.
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 8. Run Training
    print("Starting training loop...")

    # Calculate warmup steps (10% of training data)
    warmup_steps = int(len(train_dataloader) * 0.1)

    # Ensure output directory exists
    os.makedirs(Config.FINE_TUNED_MODEL_PATH, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=Config.EPOCHS,
        warmup_steps=warmup_steps,
        output_path=Config.FINE_TUNED_MODEL_PATH,
        optimizer_params={"lr": Config.LEARNING_RATE},
        weight_decay=Config.WEIGHT_DECAY,
        save_best_model=True,
        show_progress_bar=False,
    )

    print(f"Fine-tuning finished. Best model saved to: {Config.FINE_TUNED_MODEL_PATH}")
