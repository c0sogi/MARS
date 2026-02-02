import re
from library.config import CLASSES


class NumberToWords:
    """
    A self-contained helper class to convert numbers to words.
    Handles Cardinals, Ordinals, and basic digit reading.
    """

    def __init__(self):
        self.ones = [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        self.teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        self.tens = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]
        self.groups = ["", "thousand", "million", "billion"]

        self.ordinal_map = {
            "one": "first",
            "two": "second",
            "three": "third",
            "five": "fifth",
            "eight": "eighth",
            "nine": "ninth",
            "twelve": "twelfth",
        }

    def convert_cardinal(self, number):
        """Converts an integer to words."""
        if number == 0:
            return "zero"

        s = ""
        for i, name in enumerate(self.groups):
            if number == 0:
                break

            n = number % 1000
            if n != 0:
                s = (
                    self._convert_hundreds(n)
                    + (" " + name if name else "")
                    + (" " + s if s else "")
                )
            number //= 1000

        return s.strip()

    def _convert_hundreds(self, n):
        s = ""
        if n >= 100:
            s += self.ones[n // 100] + " hundred"
            n %= 100
            if n != 0:
                s += " "

        if n >= 20:
            s += self.tens[n // 10]
            n %= 10
            if n != 0:
                s += "-" + self.ones[n]
        elif n >= 10:
            s += self.teens[n - 10]
        elif n > 0:
            s += self.ones[n]
        return s

    def convert_ordinal(self, number_str):
        """Converts a number string (e.g., '1st', '23') to ordinal words."""
        # Remove suffix if present
        num_clean = re.sub(r"(st|nd|rd|th)$", "", number_str, flags=re.IGNORECASE)
        try:
            num = int(num_clean)
        except ValueError:
            return number_str  # Fallback

        cardinal = self.convert_cardinal(num)
        words = cardinal.split()
        last_word = words[-1]

        # Morph last word
        if last_word in self.ordinal_map:
            words[-1] = self.ordinal_map[last_word]
        elif last_word.endswith("y"):
            words[-1] = last_word[:-1] + "ieth"
        else:
            words[-1] = last_word + "th"

        return " ".join(words)

    def convert_digits(self, text):
        """Reads digits one by one (e.g., for phone numbers)."""
        mapping = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        return " ".join([mapping.get(c, c) for c in text])


class NormalizationRegistry:
    """
    Registry that maps semiotic classes to normalization functions.
    """

    def __init__(self):
        self.n2w = NumberToWords()

        # Currency Mapping
        self.currency_map = {
            "$": "dollar",
            "€": "euro",
            "£": "pound",
            "¥": "yen",
            "USD": "dollar",
            "EUR": "euro",
            "GBP": "pound",
            "JPY": "yen",
        }

        # Unit Mapping
        self.unit_map = {
            "kg": "kilograms",
            "km": "kilometers",
            "m": "meters",
            "cm": "centimeters",
            "mm": "millimeters",
            "g": "grams",
            "mg": "milligrams",
            "oz": "ounces",
            "lb": "pounds",
            "lbs": "pounds",
            "ml": "milliliters",
            "l": "liters",
            "%": "percent",
            "hz": "hertz",
            "khz": "kilohertz",
            "mhz": "megahertz",
            "v": "volts",
            "a": "amps",
            "w": "watts",
            "s": "seconds",
            "min": "minutes",
            "h": "hours",
            "hr": "hours",
            "ft": "feet",
            "in": "inches",
            "mi": "miles",
        }

    def normalize(self, text: str, label: str) -> str:
        """
        Main entry point to normalize a token based on its class.
        """
        if label == "PLAIN" or label == "PUNCT":
            return text

        # Dispatch to specific handler
        method_name = f"_normalize_{label.lower()}"
        method = getattr(self, method_name, None)

        if method:
            try:
                return method(text)
            except Exception:
                # Fallback to raw text if normalization logic fails
                return text

        return text

    # ==========================================
    # Specific Normalizers
    # ==========================================

    def _normalize_cardinal(self, text):
        # Remove commas
        clean_text = text.replace(",", "")
        try:
            if clean_text.isdigit() or (
                clean_text.startswith("-") and clean_text[1:].isdigit()
            ):
                return self.n2w.convert_cardinal(int(clean_text))
            # Handle Roman numerals or other formats if needed, else fallback
            return text
        except:
            return text

    def _normalize_ordinal(self, text):
        return self.n2w.convert_ordinal(text)

    def _normalize_date(self, text):
        # Heuristic for years: 1990, 2012
        if re.match(r"^\d{4}$", text):
            year = int(text)
            if year < 2000:
                # Nineteen Ninety-Nine
                first = self.n2w.convert_cardinal(year // 100)
                second = self.n2w.convert_cardinal(year % 100)
                if year % 100 < 10:
                    second = "o " + second
                return f"{first} {second}"
            elif 2000 <= year < 2010:
                # Two thousand X
                return self.n2w.convert_cardinal(year)
            else:
                # Twenty Ten
                first = self.n2w.convert_cardinal(year // 100)
                second = self.n2w.convert_cardinal(year % 100)
                if year % 100 < 10:
                    second = "o " + second
                return f"{first} {second}"

        # Simple Date: 2012-05-12 (ISO)
        # Fallback: treat as sequence of cardinals/digits
        return text.replace("-", " ").replace("/", " ")

    def _normalize_letters(self, text):
        # A.B.C -> a b c
        # Remove dots, split chars
        clean = text.replace(".", "")
        return " ".join(list(clean)).lower()

    def _normalize_digit(self, text):
        return self.n2w.convert_digits(text)

    def _normalize_decimal(self, text):
        # 12.5 -> twelve point five
        if "." in text:
            parts = text.split(".")
            if len(parts) == 2:
                integer_part = parts[0].replace(",", "")
                decimal_part = parts[1]

                int_str = (
                    self.n2w.convert_cardinal(int(integer_part))
                    if integer_part
                    else "zero"
                )
                dec_str = self.n2w.convert_digits(decimal_part)

                return f"{int_str} point {dec_str}"
        return text

    def _normalize_money(self, text):
        # $3.50, $100, £50
        # Find symbol
        symbol = None
        amount_str = text

        # Check for prefix symbols
        for sym in self.currency_map.keys():
            if text.startswith(sym):
                symbol = sym
                amount_str = text[len(sym) :]
                break

        # Check for suffix symbols (e.g. 100$)
        if not symbol:
            for sym in self.currency_map.keys():
                if text.endswith(sym):
                    symbol = sym
                    amount_str = text[: -len(sym)]
                    break

        amount_str = amount_str.replace(",", "")

        try:
            if "." in amount_str:
                parts = amount_str.split(".")
                major = int(parts[0])
                minor = int(parts[1])

                major_words = self.n2w.convert_cardinal(major)
                minor_words = self.n2w.convert_cardinal(minor)

                currency_name = self.currency_map.get(symbol, "dollar")
                if major != 1:
                    currency_name += "s"

                # Logic: "three dollars fifty cents" or "three dollars fifty"
                # Standard dataset often uses "X dollars Y cents"
                cents_name = (
                    "cents" if symbol in ["$", "USD", "€", "EUR"] else "pence"
                )  # Simplified
                if minor == 1:
                    cents_name = cents_name[:-1]  # cent/penny

                if minor > 0:
                    return f"{major_words} {currency_name} {minor_words} {cents_name}"
                else:
                    return f"{major_words} {currency_name}"
            else:
                amount = int(amount_str)
                words = self.n2w.convert_cardinal(amount)
                currency_name = self.currency_map.get(symbol, "dollar")
                if amount != 1:
                    currency_name += "s"
                return f"{words} {currency_name}"
        except:
            return text

    def _normalize_measure(self, text):
        # 10kg, 50m
        # Split number and alpha
        match = re.match(r"^([\d\.,]+)\s*([a-zA-Z%]+)$", text)
        if match:
            number_part = match.group(1)
            unit_part = match.group(2).lower()

            # Normalize number (handle decimal or cardinal)
            if "." in number_part:
                number_words = self._normalize_decimal(number_part)
                val = float(number_part.replace(",", ""))
            else:
                number_words = self._normalize_cardinal(number_part)
                val = int(number_part.replace(",", ""))

            unit_word = self.unit_map.get(unit_part, unit_part)

            # Pluralize unit if needed and not already plural/invariant
            if val != 1 and not unit_word.endswith("s") and unit_part != "hz":
                unit_word += "s"

            return f"{number_words} {unit_word}"

        return text

    def _normalize_time(self, text):
        # 12:30
        if ":" in text:
            parts = text.split(":")
            if len(parts) >= 2:
                try:
                    hour = int(parts[0])
                    minute = int(
                        parts[1][:2]
                    )  # Ignore seconds or am/pm attached immediately

                    hour_word = self.n2w.convert_cardinal(hour)
                    if minute == 0:
                        return f"{hour_word} o'clock"
                    elif minute < 10:
                        return f"{hour_word} o {self.n2w.convert_cardinal(minute)}"
                    else:
                        return f"{hour_word} {self.n2w.convert_cardinal(minute)}"
                except:
                    pass
        return text

    def _normalize_verbatim(self, text):
        return text

    def _normalize_electronic(self, text):
        # google.com -> google dot com
        # http -> h t t p
        t = text.replace(".", " dot ").replace("/", " slash ").replace(":", " colon ")
        return " ".join(t.split())  # Normalize spaces

    def _normalize_telephone(self, text):
        # Read as digits with silences
        # 123-456 -> one two three sil four five six
        t = text.replace("-", " sil ").replace("(", " ").replace(")", " ")
        words = []
        for char in t:
            if char.isdigit():
                words.append(self.n2w.convert_digits(char))
            elif char == " ":
                continue
            elif char == "s" and t[t.index(char) :].startswith("sil"):
                words.append("sil")
            else:
                words.append(char)
        # This is a bit loose, better to just read digits
        return self.n2w.convert_digits(text)
