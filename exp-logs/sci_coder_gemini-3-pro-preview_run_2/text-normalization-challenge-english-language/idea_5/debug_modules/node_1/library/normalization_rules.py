import re
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================

_ONES = {
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

_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    30: "thirtieth",
    40: "fortieth",
    50: "fiftieth",
    60: "sixtieth",
    70: "seventieth",
    80: "eightieth",
    90: "ninetieth",
}

_MONTHS = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "may": "may",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
    "jan.": "january",
    "feb.": "february",
    "mar.": "march",
    "apr.": "april",
    "jun.": "june",
    "jul.": "july",
    "aug.": "august",
    "sep.": "september",
    "oct.": "october",
    "nov.": "november",
    "dec.": "december",
    "1": "january",
    "2": "february",
    "3": "march",
    "4": "april",
    "5": "may",
    "6": "june",
    "7": "july",
    "8": "august",
    "9": "september",
    "10": "october",
    "11": "november",
    "12": "december",
    "01": "january",
    "02": "february",
    "03": "march",
    "04": "april",
    "05": "may",
    "06": "june",
    "07": "july",
    "08": "august",
    "09": "september",
}

_MEASURE_UNITS = {
    "kg": "kilograms",
    "g": "grams",
    "m": "meters",
    "cm": "centimeters",
    "mm": "millimeters",
    "km": "kilometers",
    "in": "inches",
    "ft": "feet",
    "yd": "yards",
    "mi": "miles",
    "lb": "pounds",
    "oz": "ounces",
    "l": "liters",
    "ml": "milliliters",
    "gal": "gallons",
    "s": "seconds",
    "min": "minutes",
    "h": "hours",
    "hr": "hours",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
    "a": "amperes",
    "v": "volts",
    "w": "watts",
    "kw": "kilowatts",
    "j": "joules",
    "k": "kelvin",
    "c": "celsius",
    "f": "fahrenheit",
    "%": "percent",
    "sq": "square",
    "ha": "hectares",
}

_CURRENCY_MAP = {
    "$": "dollars",
    "£": "pounds",
    "€": "euros",
    "¥": "yen",
    "us$": "dollars",
    "usd": "dollars",
    "aud": "dollars",
    "cad": "dollars",
    "gbp": "pounds",
    "eur": "euros",
    "inr": "rupees",
}

_CURRENCY_MINOR = {
    "$": "cents",
    "£": "pence",
    "€": "cents",
    "us$": "cents",
    "usd": "cents",
    "aud": "cents",
    "cad": "cents",
    "gbp": "pence",
    "eur": "cents",
}


# ==========================================
# Core Helper Functions
# ==========================================


def _num2words(num_str: str) -> str:
    """Converts a string integer to words (e.g., '123' -> 'one hundred twenty-three')."""
    try:
        n = int(num_str.replace(",", ""))
    except ValueError:
        return num_str

    if n < 0:
        return "minus " + _num2words(str(abs(n)))
    if n == 0:
        return "zero"

    parts = []

    # Billions
    if n >= 1_000_000_000:
        billions = n // 1_000_000_000
        parts.append(_num2words(str(billions)) + " billion")
        n %= 1_000_000_000

    # Millions
    if n >= 1_000_000:
        millions = n // 1_000_000
        parts.append(_num2words(str(millions)) + " million")
        n %= 1_000_000

    # Thousands
    if n >= 1_000:
        thousands = n // 1_000
        parts.append(_num2words(str(thousands)) + " thousand")
        n %= 1_000

    # Hundreds
    if n >= 100:
        hundreds = n // 100
        parts.append(_ONES[hundreds] + " hundred")
        n %= 100

    # Tens and Ones
    if n > 0:
        if n in _ONES:
            parts.append(_ONES[n])
        elif n in _TENS:
            parts.append(_TENS[n])
        else:
            tens = (n // 10) * 10
            ones = n % 10
            parts.append(f"{_TENS[tens]} {_ONES[ones]}")

    return " ".join(parts)


def _ordinal2words(num_str: str) -> str:
    """Converts a string integer to ordinal words (e.g., '21' -> 'twenty-first')."""
    # Remove suffixes like st, nd, rd, th if present
    clean_num = re.sub(r"(st|nd|rd|th)$", "", num_str, flags=re.IGNORECASE)
    try:
        n = int(clean_num.replace(",", ""))
    except ValueError:
        return num_str

    if n in _ORDINALS:
        return _ORDINALS[n]

    # For larger numbers, convert the bulk to cardinal and the last part to ordinal
    cardinal_str = _num2words(str(n))
    words = cardinal_str.split()
    last_word = words[-1]

    # Mapping common last words to ordinals
    # e.g. "one" -> "first", "eight" -> "eighth"
    # Simple heuristic: reverse lookup or manual replacement
    # Since _num2words output is controlled, we can map the end.

    replacements = {
        "one": "first",
        "two": "second",
        "three": "third",
        "five": "fifth",
        "eight": "eighth",
        "nine": "ninth",
        "twelve": "twelfth",
    }

    # Handle "twenty", "thirty" -> "twentieth"
    if last_word.endswith("y"):
        words[-1] = last_word[:-1] + "ieth"
    elif last_word in replacements:
        words[-1] = replacements[last_word]
    else:
        words[-1] = last_word + "th"

    return " ".join(words)


def _spell_out(text: str) -> str:
    """Spells out text character by character (e.g., 'FBI' -> 'f b i')."""
    chars = []
    for c in text:
        if c.isalnum():
            chars.append(c.lower())
        else:
            # Map common symbols
            if c == ".":
                chars.append("dot")
            elif c == "/":
                chars.append("slash")
            elif c == "-":
                chars.append("dash")
    return " ".join(chars)


# ==========================================
# Class-Specific Handlers
# ==========================================


def _expand_cardinal(text: str) -> str:
    # Handle digits with commas or simple integers
    if re.match(r"^-?\d{1,3}(,\d{3})*(\.0+)?$", text) or re.match(r"^-?\d+$", text):
        return _num2words(text)
    # Fallback for Roman numerals or mixed
    return text


def _expand_ordinal(text: str) -> str:
    return _ordinal2words(text)


def _expand_date(text: str) -> str:
    # Year logic: 1990 -> nineteen ninety, 2000 -> two thousand
    if re.match(r"^\d{4}$", text):
        year = int(text)
        if 2000 <= year <= 2009:
            return _num2words(text)
        elif 1000 <= year <= 9999:
            prefix = int(text[:2])
            suffix = int(text[2:])
            p_str = _num2words(str(prefix))
            if suffix == 0:
                return f"{p_str} hundred"
            elif suffix < 10:
                return f"{p_str} o {_num2words(str(suffix))}"
            else:
                return f"{p_str} {_num2words(str(suffix))}"

    # Simple YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        parts = text.split("-")
        y, m, d = parts[0], parts[1], parts[2]
        m_str = _MONTHS.get(m, _num2words(m))
        d_str = _ordinal2words(d)
        y_str = _expand_date(y)
        return f"the {d_str} of {m_str} {y_str}"

    # DD/MM/YYYY or MM/DD/YYYY - ambiguous, usually context dependent.
    # Defaulting to treating as cardinal sequence or spell out if complex.
    return _spell_out(text)


def _expand_money(text: str) -> str:
    # Pattern: Currency symbol + amount
    # e.g., $3.50, £10, USD 500
    match = re.match(
        r"^([^\d\s]+)?\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)?([^\d\s]+)?$", text
    )
    if not match:
        return text

    prefix, amount, suffix = match.groups()
    currency = prefix if prefix else (suffix if suffix else "$")
    amount = amount if amount else "0"

    # Normalize currency name
    curr_name = _CURRENCY_MAP.get(currency.lower(), "dollars")
    minor_name = _CURRENCY_MINOR.get(currency.lower(), "cents")

    if "." in amount:
        major, minor = amount.split(".", 1)
        major_val = int(major.replace(",", ""))
        # Handle "3.5" as "3.50" -> 50 cents, but "3.05" -> 5 cents
        if len(minor) == 1:
            minor += "0"
        minor = minor[:2]  # Truncate to 2 decimals for cents
        minor_val = int(minor)

        res = f"{_num2words(str(major_val))} {curr_name}"
        if major_val == 1:
            res = res.replace(curr_name, curr_name[:-1])  # singular

        if minor_val > 0:
            res += f", {_num2words(str(minor_val))} {minor_name}"
            if minor_val == 1:
                res = res.replace(minor_name, minor_name[:-1])
        return res
    else:
        val = int(amount.replace(",", ""))
        res = f"{_num2words(str(val))} {curr_name}"
        if val == 1:
            res = res.replace(curr_name, curr_name[:-1])
        return res


def _expand_measure(text: str) -> str:
    # Split number and unit (e.g., 5kg, 5 kg)
    match = re.match(r"^(-?\d+(?:\.\d+)?)\s?([a-zA-Z%]+)$", text)
    if match:
        num_str, unit = match.groups()

        # Expand number (can be decimal)
        if "." in num_str:
            num_exp = _expand_decimal(num_str)
            is_plural = True  # 1.5 kilograms
        else:
            num_exp = _num2words(num_str)
            is_plural = float(num_str) != 1

        unit_exp = _MEASURE_UNITS.get(unit.lower(), unit)

        # Handle singular/plural for unit
        if (
            not is_plural
            and unit_exp.endswith("s")
            and unit_exp not in ["celsius", "siemens"]
        ):
            unit_exp = unit_exp[:-1]

        return f"{num_exp} {unit_exp}"

    return text


def _expand_decimal(text: str) -> str:
    if "." not in text:
        return _expand_cardinal(text)

    parts = text.split(".")
    integer_part = _num2words(parts[0]) if parts[0] else "zero"

    # Fractional part is read digit by digit
    fractional_part = " ".join([_ONES[int(d)] for d in parts[1]])

    return f"{integer_part} point {fractional_part}"


def _expand_digit(text: str) -> str:
    # "0800" -> "zero eight zero zero"
    return " ".join([_ONES[int(c)] for c in text if c.isdigit()])


def _expand_letters(text: str) -> str:
    # "U.S.A." -> "u s a"
    # "FBI" -> "f b i"
    clean = text.replace(".", "")
    return " ".join([c.lower() for c in clean])


def _expand_time(text: str) -> str:
    # 12:30, 8:00
    if ":" in text:
        parts = text.split(":")
        h = int(parts[0])
        m = int(parts[1])

        h_str = _num2words(str(h))
        if m == 0:
            return f"{h_str} o'clock"
        elif m < 10:
            return f"{h_str} o {_num2words(str(m))}"
        else:
            return f"{h_str} {_num2words(str(m))}"
    return text


def _expand_fraction(text: str) -> str:
    # 1/2, 3/4
    if "/" in text:
        num, den = text.split("/")
        num_str = _num2words(num)
        den_str = _ordinal2words(den)
        if int(den) == 2:
            den_str = "half"
        if int(den) == 4:
            den_str = "quarter"

        res = f"{num_str} {den_str}"
        if int(num) > 1:
            if den_str == "half":
                res = res.replace("half", "halves")
            else:
                res += "s"
        return res
    return text


def _expand_verbatim(text: str) -> str:
    # Symbols
    replacements = {"&": "and", "@": "at", "#": "hash", "%": "percent"}
    return replacements.get(text, text)


# ==========================================
# Main Dispatcher
# ==========================================


def normalize_token(text: str, label: str) -> str:
    """
    Converts a raw token into normalized text based on the predicted label.

    Args:
        text (str): The raw token text (e.g., "$3.50").
        label (str): The predicted class label (e.g., "MONEY").

    Returns:
        str: The normalized spoken form (e.g., "three dollars, fifty cents").
    """
    # Safety check for empty or non-string
    if not text or not isinstance(text, str):
        return str(text)

    # Dispatcher
    try:
        if label == "PLAIN":
            return text
        elif label == "PUNCT":
            return text
        elif label == "CARDINAL":
            return _expand_cardinal(text)
        elif label == "DATE":
            return _expand_date(text)
        elif label == "LETTERS":
            return _expand_letters(text)
        elif label == "VERBATIM":
            return _expand_verbatim(text)
        elif label == "MEASURE":
            return _expand_measure(text)
        elif label == "ORDINAL":
            return _expand_ordinal(text)
        elif label == "DECIMAL":
            return _expand_decimal(text)
        elif label == "MONEY":
            return _expand_money(text)
        elif label == "DIGIT":
            return _expand_digit(text)
        elif label == "TIME":
            return _expand_time(text)
        elif label == "FRACTION":
            return _expand_fraction(text)
        elif label == "TELEPHONE":
            # Treat telephone as sequence of digits/silence
            return _expand_digit(text)  # Simplified
        elif label == "ELECTRONIC":
            # URLs etc.
            return _spell_out(text)
        elif label == "ADDRESS":
            # Address usually contains cardinals and letters mixed
            return _expand_cardinal(text)  # Fallback to cardinal for house numbers
        else:
            return text
    except Exception:
        # Fail-safe: if any logic crashes, return original text or simple cleaned version
        return text
