import re
from typing import List, Tuple, Optional
from library.config import ModelConfig
from library.text_processing import TextPreprocessor


class ShortAnswerHeuristic:
    """
    Implements heuristics for extracting short and Yes/No answers
    from a predicted long answer text.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        # Regex for splitting sentences:
        # Look for punctuation (.?!) followed by whitespace.
        # Negative lookbehinds attempt to avoid splitting on common abbreviations (e.g., U.S., Mr.).
        self.sentence_pattern = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
        )
        self.threshold = ModelConfig.SHORT_CONFIDENCE_THRESHOLD

    def split_sentences(self, text: str) -> List[str]:
        """
        Segments text into individual sentences using regex heuristics.

        Args:
            text (str): The long answer text to split.

        Returns:
            List[str]: A list of sentence strings.
        """
        if not text:
            return []

        # Split and filter empty strings
        sentences = [s.strip() for s in self.sentence_pattern.split(text) if s.strip()]

        # Fallback: if regex didn't split but text exists, return as single sentence
        if not sentences and text.strip():
            return [text.strip()]

        return sentences

    def _compute_jaccard(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Computes Jaccard similarity between two lists of tokens.
        J(A, B) = |A intersection B| / |A union B|
        """
        set1 = set(tokens1)
        set2 = set(tokens2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def find_best_sentence(
        self, question_text: str, long_answer_text: str
    ) -> Tuple[Optional[str], float]:
        """
        Identifies the sentence within the long answer that best matches the question
        based on Jaccard similarity of preprocessed tokens.

        Args:
            question_text (str): The input question.
            long_answer_text (str): The predicted long answer text.

        Returns:
            Tuple[Optional[str], float]:
                - The best sentence text (or None if the score is below SHORT_CONFIDENCE_THRESHOLD).
                - The similarity score of that sentence.
        """
        if not long_answer_text or not question_text:
            return None, 0.0

        sentences = self.split_sentences(long_answer_text)

        # Preprocess question once (tokenize, remove stops, stem)
        q_tokens = self.preprocessor.preprocess(question_text)

        best_score = -1.0
        best_sentence = None

        for sentence in sentences:
            # Preprocess candidate sentence
            s_tokens = self.preprocessor.preprocess(sentence)

            score = self._compute_jaccard(q_tokens, s_tokens)

            if score > best_score:
                best_score = score
                best_sentence = sentence

        # Apply confidence threshold
        if best_score >= self.threshold:
            return best_sentence, best_score
        else:
            return None, best_score

    def check_yes_no(self, short_answer_text: str) -> str:
        """
        Determines if the short answer implies a YES or NO answer.
        Checks the first token of the short answer text.

        Args:
            short_answer_text (str): The selected short answer text.

        Returns:
            str: 'YES', 'NO', or 'NONE'.
        """
        if not short_answer_text:
            return "NONE"

        # Use tokenize only (no stemming) to preserve "yes"/"no" exact spelling
        # Tokenizer converts to lowercase
        tokens = self.preprocessor.tokenize(short_answer_text)

        if not tokens:
            return "NONE"

        first_token = tokens[0]

        if first_token == "yes":
            return "YES"
        elif first_token == "no":
            return "NO"

        return "NONE"
