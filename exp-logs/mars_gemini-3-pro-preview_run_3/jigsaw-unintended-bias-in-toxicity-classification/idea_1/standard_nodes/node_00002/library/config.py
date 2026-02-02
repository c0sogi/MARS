import os


class Config:
    """
    Global configuration for the Toxicity Classification project.
    """

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Metadata paths (pre-split data)
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"

    # Output directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Specific file paths for outputs
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "nbow_model.pth")
    VOCAB_SAVE_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    TEXT_COL = "comment_text"
    TARGET_COL = "target"
    ID_COL = "id"

    # Identity columns available in the dataset for metric calculation
    IDENTITY_COLUMNS = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]

    # Vocabulary settings
    VOCAB_SIZE = 60000  # Number of most frequent words to keep
    MAX_LEN = 250  # Max sequence length for padding/truncation (if needed)

    # ==========================================
    # Model Hyperparameters (NBOW)
    # ==========================================
    EMBED_DIM = 128
    HIDDEN_DIM = 128
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 512
    LEARNING_RATE = 2e-3
    NUM_EPOCHS = 15
    PATIENCE = 4  # Early stopping patience

    # ==========================================
    # Bias Mitigation Strategy: Stochastic Identity Masking
    # ==========================================
    # Probability to mask an identity term during training
    IDENTITY_MASK_PROB = 0.5

    # Mapping of identity categories to specific keywords found in text.
    # These terms will be candidates for masking to prevent overfitting to identity markers.
    IDENTITY_KEYWORDS = {
        "male": [
            "male",
            "man",
            "men",
            "boy",
            "boys",
            "father",
            "dad",
            "husband",
            "brother",
            "son",
            "males",
        ],
        "female": [
            "female",
            "woman",
            "women",
            "girl",
            "girls",
            "mother",
            "mom",
            "wife",
            "sister",
            "daughter",
            "females",
        ],
        "homosexual_gay_or_lesbian": [
            "gay",
            "lesbian",
            "homosexual",
            "queer",
            "lgbt",
            "lgbtq",
            "bisexual",
            "transgender",
            "trans",
        ],
        "christian": [
            "christian",
            "christians",
            "catholic",
            "protestant",
            "church",
            "bible",
            "jesus",
            "christ",
        ],
        "jewish": ["jewish", "jew", "jews", "judaism", "synagogue", "torah", "semitic"],
        "muslim": ["muslim", "muslims", "islam", "islamic", "allah", "quran", "mosque"],
        "black": ["black", "blacks", "african"],
        "white": ["white", "whites", "caucasian", "european"],
        "psychiatric_or_mental_illness": [
            "mental",
            "illness",
            "depression",
            "bipolar",
            "schizophrenia",
            "autism",
            "autistic",
            "retarded",
            "insane",
            "crazy",
            "psycho",
        ],
    }

    @classmethod
    def get_identity_term_set(cls):
        """
        Returns a flattened set of all identity keywords for efficient lookup.
        """
        term_set = set()
        for keywords in cls.IDENTITY_KEYWORDS.values():
            for word in keywords:
                term_set.add(word)
        return term_set
