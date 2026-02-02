import os
import torch


class GlobalConfig:
    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    TRAIN_METADATA_PATH = os.path.join(INPUT_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(INPUT_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(INPUT_DIR, "test_metadata.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # Debugging
    DEBUG = False
    DEBUG_SAMPLES = 100  # Number of samples to use when DEBUG is True

    # --------------------------------------------------------------------------
    # Target Columns (30 Labels)
    # --------------------------------------------------------------------------
    TARGET_COLS = [
        "question_asker_intent_understanding",
        "question_body_critical",
        "question_conversational",
        "question_expect_short_answer",
        "question_fact_seeking",
        "question_has_commonly_accepted_answer",
        "question_interestingness_others",
        "question_interestingness_self",
        "question_multi_intent",
        "question_not_really_a_question",
        "question_opinion_seeking",
        "question_type_choice",
        "question_type_compare",
        "question_type_consequence",
        "question_type_definition",
        "question_type_entity",
        "question_type_instructions",
        "question_type_procedure",
        "question_type_reason_explanation",
        "question_type_spelling",
        "question_well_written",
        "answer_helpful",
        "answer_level_of_information",
        "answer_plausible",
        "answer_relevance",
        "answer_satisfaction",
        "answer_type_instructions",
        "answer_type_procedure",
        "answer_type_reason_explanation",
        "answer_well_written",
    ]


class ModelConfig:
    """
    Configuration for a specific model stream (e.g., MPNet or RoBERTa).
    """

    def __init__(
        self, model_name, name_tag, max_length=512, batch_size=8, lr=2e-5, epochs=5
    ):
        self.model_name = model_name
        self.name_tag = name_tag

        # Hyperparameters
        self.max_length = max_length
        self.train_batch_size = batch_size
        self.valid_batch_size = batch_size * 2
        self.learning_rate = lr
        self.epochs = epochs
        self.hidden_dropout_prob = 0.1
        self.weight_decay = 0.01

        # Output Paths
        self.output_dir = os.path.join(GlobalConfig.WORKING_DIR, name_tag)
        os.makedirs(self.output_dir, exist_ok=True)

        # Checkpoints
        self.model_save_path = os.path.join(
            self.output_dir, f"{name_tag}_finetuned.pth"
        )

        # Cached Features (Numpy)
        self.train_features_path = os.path.join(
            self.output_dir, f"{name_tag}_train_features.npy"
        )
        self.val_features_path = os.path.join(
            self.output_dir, f"{name_tag}_val_features.npy"
        )
        self.test_features_path = os.path.join(
            self.output_dir, f"{name_tag}_test_features.npy"
        )

        # Ridge Model
        self.ridge_path = os.path.join(
            self.output_dir, f"{name_tag}_ridge_model.joblib"
        )


# --------------------------------------------------------------------------
# Stream Configurations
# --------------------------------------------------------------------------

# Stream 1: MPNet Cross-Encoder
MPNET_CONFIG = ModelConfig(
    model_name="sentence-transformers/all-mpnet-base-v2",
    name_tag="mpnet",
    max_length=512,
    batch_size=8,
    lr=2e-5,
    epochs=5,
)

# Stream 2: RoBERTa Cross-Encoder
ROBERTA_CONFIG = ModelConfig(
    model_name="roberta-base",
    name_tag="roberta",
    max_length=512,
    batch_size=8,
    lr=2e-5,
    epochs=5,
)
