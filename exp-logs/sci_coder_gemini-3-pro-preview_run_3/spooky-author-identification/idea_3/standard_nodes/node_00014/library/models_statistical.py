import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from library.config import Config


class StatisticalExpert:
    """
    The Statistical Branch (Surface Expert) of the ensemble.
    Combines Word and Character N-grams via TF-IDF with a Voting Classifier
    (Logistic Regression + Multinomial Naive Bayes).
    """

    def __init__(self):
        # 1. Feature Extraction: TF-IDF on Word N-grams
        # Captures lexical usage and phrases
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=Config.WORD_NGRAM_RANGE,
            max_features=Config.TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            strip_accents="unicode",
        )

        # 2. Feature Extraction: TF-IDF on Character N-grams
        # Captures morphological patterns and punctuation style
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=Config.CHAR_NGRAM_RANGE,
            max_features=Config.TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            strip_accents="unicode",
        )

        # Combine Word and Char features into a single sparse matrix
        self.preprocessor = FeatureUnion(
            [("word_tfidf", self.word_vectorizer), ("char_tfidf", self.char_vectorizer)]
        )

        # 3. Classifiers
        # Logistic Regression: Robust baseline for high-dimensional sparse data
        self.clf_lr = LogisticRegression(
            solver="liblinear",
            multi_class="ovr",
            C=1.0,
            random_state=Config.SEED,
            max_iter=1000,
        )

        # Multinomial Naive Bayes: Effective for text classification
        self.clf_nb = MultinomialNB(alpha=0.01)

        # Voting Ensemble: Soft voting averages the predicted probabilities
        # Weighted 3:1 towards LR as per Lesson 3 (Discriminative vs Generative)
        self.ensemble = VotingClassifier(
            estimators=[("lr", self.clf_lr), ("nb", self.clf_nb)],
            voting="soft",
            weights=[3, 1],
            n_jobs=1,
        )

        # Final Pipeline construction
        self.model = Pipeline(
            [("preprocessor", self.preprocessor), ("ensemble", self.ensemble)]
        )

    def fit(self, X, y):
        """
        Fits the statistical expert model pipeline.

        Args:
            X: Iterable of text samples (e.g., pandas Series or list of strings).
            y: Iterable of target labels.

        Returns:
            self: The fitted instance.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input text.

        Args:
            X: Iterable of text samples.

        Returns:
            np.ndarray: Probability array of shape (n_samples, n_classes).
                        Columns correspond to the sorted classes (EAP, HPL, MWS).
        """
        return self.model.predict_proba(X)
