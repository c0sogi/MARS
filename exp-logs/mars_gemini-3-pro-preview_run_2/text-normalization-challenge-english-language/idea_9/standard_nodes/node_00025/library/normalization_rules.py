import re
from library.config import Config

# ==========================================
# 1. Core Number Conversion Logic
# ==========================================


class NumberConverter:
    """
    A standalone implementation of number-to-text conversion to avoid
    dependencies on external libraries like num2words or inflect.
    """

    ONES = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
    }

    TENS = {
        2: "twenty",
        3: "thirty",
        4: "forty",
        5: "fifty",
        6: "sixty",
        7: "seventy",
        8: "eighty",
        9: "ninety",
    }

    ORDINAL_MAP = {
        "one": "first",
        "two": "second",
        "three": "third",
        "five": "fifth",
        "eight": "eighth",
        "nine": "ninth",
        "twelve": "twelfth",
    }

    @classmethod
    def integer_to_words(cls, n: int) -> str:
        """Converts an integer to its cardinal word representation."""
        if n < 0:
            return "minus " + cls.integer_to_words(-n)
        if n == 0:
            return "zero"

        parts = []

        # Billions
        if n >= 1_000_000_000:
            billions = n // 1_000_000_000
            parts.append(cls.integer_to_words(billions) + " billion")
            n %= 1_000_000_000

        # Millions
        if n >= 1_000_000:
            millions = n // 1_000_000
            parts.append(cls.integer_to_words(millions) + " million")
            n %= 1_000_000

        # Thousands
        if n >= 1_000:
            thousands = n // 1_000
            parts.append(cls.integer_to_words(thousands) + " thousand")
            n %= 1_000

        # Hundreds
        if n >= 100:
            hundreds = n // 100
            parts.append(cls.ONES[hundreds] + " hundred")
            n %= 100

        # Tens and Ones
        if n > 0:
            if n < 20:
                parts.append(cls.ONES[n])
            else:
                tens = n // 10
                ones = n % 10
                text = cls.TENS[tens]
                if ones > 0:
                    text += " " + cls.ONES[ones]
                parts.append(text)

        return " ".join(parts)

    @classmethod
    def integer_to_ordinal(cls, n: int) -> str:
        """Converts an integer to its ordinal word representation."""
        # Handle specific raw string suffixes if passed elsewhere, but here we take int.
        # Logic: Convert to cardinal, then morph the last word.

        cardinal_str = cls.integer_to_words(n)
        words = cardinal_str.split()
        last_word = words[-1]

        # Morph last word
        if last_word in cls.ORDINAL_MAP:
            words[-1] = cls.ORDINAL_MAP[last_word]
        elif last_word.endswith("y"):
            words[-1] = last_word[:-1] + "ieth"
        else:
            words[-1] = last_word + "th"

        return " ".join(words)

    @classmethod
    def digit_to_word(cls, digit_char: str) -> str:
        """Converts a single digit character to word."""
        if digit_char.isdigit():
            return cls.ONES[int(digit_char)]
        return digit_char


# ==========================================
# 2. Class-Specific Handlers
# ==========================================


def _clean_number_string(text: str) -> str:
    """Removes commas from number strings."""
    return text.replace(",", "")


def convert_plain(text: str) -> str:
    """Returns text as is."""
    return text


def convert_punct(text: str) -> str:
    """Returns punctuation as is."""
    return text


def convert_cardinal(text: str) -> str:
    """
    Converts cardinal numbers to words.
    Example: "1,234" -> "one thousand two hundred thirty four"
    """
    clean_text = _clean_number_string(text)

    # Check if it's a Roman Numeral (basic check)
    if re.match(r"^[IVXLCDM]+$", clean_text):
        # Fallback: usually Roman numerals in normalization are read as letters
        # or cardinals depending on context. In this deterministic path,
        # if it's purely alphabetic, it might be misclassified letters or actual roman.
        # Given the complexity, we treat as letters if it fails int conversion.
        try:
            # Attempt int conversion just in case
            val = int(clean_text)
            return NumberConverter.integer_to_words(val)
        except ValueError:
            # Treat as letters
            return convert_letters(text)

    try:
        val = int(clean_text)
        return NumberConverter.integer_to_words(val)
    except ValueError:
        # Fallback for non-integer cardinals (should be rare in this class)
        return text


def convert_ordinal(text: str) -> str:
    """
    Converts ordinal numbers to words.
    Example: "1st" -> "first", "23rd" -> "twenty third"
    """
    # Remove suffix (st, nd, rd, th) to get the number
    clean_text = _clean_number_string(text)
    match = re.match(r"^(-?\d+)(st|nd|rd|th)?$", clean_text, re.IGNORECASE)

    if match:
        number_str = match.group(1)
        try:
            val = int(number_str)
            return NumberConverter.integer_to_ordinal(val)
        except ValueError:
            return text

    # Roman Ordinals (e.g. "II") are hard without context, return as is or letters
    return text


def convert_digit(text: str) -> str:
    """
    Converts a sequence of digits to words, digit by digit.
    Example: "07" -> "zero seven"
    """
    # Sometimes DIGIT class contains things like "123".
    # Standard normalization for DIGIT is usually reading digits one by one.
    words = []
    for char in text:
        if char.isdigit():
            words.append(NumberConverter.digit_to_word(char))
        else:
            # If there are symbols, keep them or map them?
            # Usually DIGIT implies pure digits. If not, just append char.
            words.append(char)
    return " ".join(words)


def convert_decimal(text: str) -> str:
    """
    Converts decimal numbers.
    Example: "3.14" -> "three point one four"
    """
    clean_text = _clean_number_string(text)

    # Split by the first period
    if "." in clean_text:
        parts = clean_text.split(".", 1)
        integer_part = parts[0]
        fractional_part = parts[1]

        result = []

        # Integer part -> Cardinal
        if integer_part:
            try:
                result.append(NumberConverter.integer_to_words(int(integer_part)))
            except ValueError:
                result.append(integer_part)  # Fallback
        else:
            # ".5" -> "point five"? Or "zero point five"?
            # Usually implies zero if missing.
            pass

        result.append("point")

        # Fractional part -> Digit by Digit
        fractional_words = []
        for char in fractional_part:
            if char.isdigit():
                fractional_words.append(NumberConverter.digit_to_word(char))
            else:
                fractional_words.append(char)
        result.append(" ".join(fractional_words))

        return " ".join(result)
    else:
        # No decimal point? Treat as cardinal.
        return convert_cardinal(text)


def convert_letters(text: str) -> str:
    """
    Converts letter sequences/acronyms.
    Example: "FBI" -> "f b i"
    """
    # Remove periods if they are separators like "U.S.A." -> "U S A"
    # But be careful not to remove all punctuation if it's meaningful.
    # Standard normalization: "U.S.A." -> "u s a"

    clean_text = text
    if clean_text.endswith("."):
        clean_text = clean_text[:-1]

    # Replace dots with spaces
    clean_text = clean_text.replace(".", " ")

    # Split into characters, filter empty, lowercase
    chars = [c.lower() for c in clean_text if c.isalnum()]

    return " ".join(chars)


# ==========================================
# 3. Dispatch Logic
# ==========================================


def apply_rule(text: str, label: str) -> str:
    """
    Applies the deterministic normalization rule based on the class label.

    Args:
        text (str): The raw input token.
        label (str): The predicted class label.

    Returns:
        str: The normalized text.
    """
    # Ensure label is valid
    if label not in Config.ALL_CLASSES:
        # Fallback to PLAIN if unknown class
        return convert_plain(text)

    if label == "PLAIN":
        return convert_plain(text)
    elif label == "PUNCT":
        return convert_punct(text)
    elif label == "CARDINAL":
        return convert_cardinal(text)
    elif label == "ORDINAL":
        return convert_ordinal(text)
    elif label == "DIGIT":
        return convert_digit(text)
    elif label == "LETTERS":
        return convert_letters(text)
    elif label == "DECIMAL":
        return convert_decimal(text)
    else:
        # If a Path B class is passed here by mistake, we return the raw text
        # or handle it as PLAIN. Ideally, the router shouldn't send Path B here.
        return text
