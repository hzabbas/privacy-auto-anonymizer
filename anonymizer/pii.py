"""
English-first PII (Personally Identifiable Information) recognition over OCR text.

Design (see RESEARCH_PAPERS.md):
  * Pattern recognizers with checksum / structural validation, following the
    recognizer architecture of Microsoft Presidio (regex + validation + context words).
      - Credit cards are validated with the Luhn checksum (ISO/IEC 7812).
      - IBANs are validated with the ISO 13616 mod-97 checksum.
      - US SSNs are validated against SSA allocation rules.
      - Phone numbers use Google's libphonenumber (``phonenumbers``) plus a NANP pattern.
  * Contextual "label: value" recognizers redact only the *value* that follows a
    sensitive label (password, account no, passport, ...). Plain keywords inside prose
    (e.g. "maintaining confidential records") are NOT sensitive on their own; flagging
    them was the main source of false positives in the previous version.
  * Named-entity recognition (PERSON / LOCATION) with a fine-tuned BERT NER model
    (dslim/bert-base-NER, CoNLL-2003), loaded lazily and optional.

All recognizers operate on a single string and return character spans, so the
caller can map spans back onto OCR word boxes.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class PIISpan:
    start: int
    end: int
    entity: str
    score: float
    source: str = "pattern"


# Human readable labels (shown in the UI) and merge priority (higher wins).
# IP_ADDRESS outranks PHONE because a dotted quad is far more specific than the
# (deliberately lenient) libphonenumber matcher that can also read it as a phone.
ENTITY_INFO = {
    "CREDIT_CARD": ("Credit card", 100),
    "IBAN": ("IBAN", 100),
    "SSN": ("SSN", 100),
    # A full MRZ line outranks everything inside it: the line itself is the
    # machine-readable copy of the document's identity fields.
    "MRZ": ("MRZ", 98),
    "SECRET": ("Secret / key", 95),
    "EMAIL": ("Email", 90),
    "IP_ADDRESS": ("IP address", 88),
    "PHONE": ("Phone", 85),
    "ID_NUMBER": ("ID number", 80),
    "DATE_OF_BIRTH": ("Date of birth", 75),
    "DATE": ("Date", 70),
    "ADDRESS": ("Address", 65),
    "URL": ("URL", 60),
    "PERSON": ("Person name", 55),
    "LOCATION": ("Location", 50),
}


def entity_label(entity: str) -> str:
    return ENTITY_INFO.get(entity, (entity.replace("_", " ").title(), 0))[0]


def _priority(entity: str) -> int:
    return ENTITY_INFO.get(entity, ("", 0))[1]


# --------------------------------------------------------------------------- #
# Text normalisation (length preserving, so offsets stay valid)
# --------------------------------------------------------------------------- #

# Persian / Arabic-Indic digits -> ASCII (1:1 character mapping).
_DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"
)

# Common OCR confusions inside numeric strings.
_OCR_DIGIT_CONFUSIONS = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1", "i": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
    "G": "6", "b": "6",
}


def _numeric_view(text: str) -> str:
    """
    Returns a same-length copy of ``text`` where letters that are typical OCR
    confusions for digits (O->0, l->1, S->5 ...) are replaced *only* when they sit
    between digits, e.g. "55O-12l4" -> "550-1214". Plain words are never touched.
    """
    chars = list(text)
    n = len(chars)

    def neighbour_is_digit(idx: int, step: int) -> bool:
        j = idx + step
        # Skip a single separator
        if 0 <= j < n and chars[j] in " -./()":
            j += step
        return 0 <= j < n and chars[j].isdigit()

    for i, ch in enumerate(chars):
        if ch in _OCR_DIGIT_CONFUSIONS and neighbour_is_digit(i, -1) and neighbour_is_digit(i, 1):
            chars[i] = _OCR_DIGIT_CONFUSIONS[ch]
    return "".join(chars)


_PHONE_SEPARATORS = " -./()"

def _looks_like_ipv4(surface: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", surface))


def _plausible_phone_match(surface: str) -> bool:
    """
    Guards the libphonenumber matcher, which is deliberately lenient: with the
    standard region set it readily reads a 10-digit national-ID run
    ("0499370899" as a German number) or a dotted IPv4 address as a phone number.

    Real phone numbers in a document are written internationally (leading '+'),
    carry separators, or are long enough to include a country/area prefix.
    """
    surface = surface.strip()
    if not surface or _looks_like_ipv4(surface):
        return False
    if surface.startswith("+"):
        return True
    if any(ch in _PHONE_SEPARATORS for ch in surface[1:]):
        return True
    return len(_digits(surface)) >= 11


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #

def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_valid(number: str) -> bool:
    digits = [int(d) for d in _digits(number)]
    if len(digits) < 12:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def iban_valid(iban: str) -> bool:
    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def ssn_valid(area: str, group: str, serial: str) -> bool:
    if area in ("000", "666") or area.startswith("9"):
        return False
    if group == "00" or serial == "0000":
        return False
    # Well-known invalid / advertising SSNs
    if f"{area}{group}{serial}" in ("123456789", "078051120", "219099999"):
        return False
    return True


def iranian_national_id_valid(code: str) -> bool:
    if len(code) != 10 or not code.isdigit():
        return False
    # Many repeating identical digits are usually invalid
    if len(set(code)) == 1:
        return False
    check = sum(int(code[i]) * (10 - i) for i in range(9)) % 11
    last_digit = int(code[9])
    if check < 2:
        return last_digit == check
    else:
        return last_digit == 11 - check


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in freq.values())


# --------------------------------------------------------------------------- #
# Pattern recognizers
# --------------------------------------------------------------------------- #

# NOTE: separators deliberately exclude "\n" so that no pattern can span two OCR lines.
_SEP = r"[ \t.\-]"

EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*[ \t]?@[ \t]?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}(?![\w])"
)
# OCR frequently drops or misreads the dots of an address ("anthony@caldwellcom",
# "jane@acme,co"). Any local@domain token is still an e-mail / handle -> PII.
EMAIL_OCR_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.,\-]*[A-Za-z0-9]"
)

NANP_PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+?1[ \t.\-]?)?\(?[2-9]\d{2}\)?[ \t.\-]?\d{3}[ \t.\-]?\d{4}(?:[ \t]*(?:x|ext\.?)[ \t]*\d{1,5})?(?!\w)"
)
INTL_PHONE_RE = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ \t.\-]?\(?\d{1,4}\)?){2,5}(?!\w)")
# Explicitly formatted 3-3-4 numbers ("(159) 357-8426", "159.357.8426"). The formatting
# itself is strong evidence, so the NANP area-code rule is not enforced here.
FORMATTED_PHONE_RE = re.compile(
    r"(?<![\w])(?:\(\d{3}\)[ \t.\-]?\d{3}[ \t.\-]\d{4}|\d{3}([.\-])\d{3}\1\d{4})(?![\w\-])"
)

SSN_RE = re.compile(r"(?<![\d\-])(\d{3})[- ](\d{2})[- ](\d{4})(?![\d\-])")
SSN_CONTEXT_RE = re.compile(
    r"(?i)\b(?:ssn|social[ \t]+security(?:[ \t]+(?:no|number|#))?)\b[^\n\d]{0,12}(\d{9})(?!\d)"
)

CREDIT_CARD_RE = re.compile(r"(?<![\d\-])(?:\d[ \t\-]?){12,18}\d(?![\d\-])")

IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ \t]?[A-Z0-9]{2,4}){3,8}\b")

IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']{3,}")

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_PATTERN = (
    r"(?:\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
    r"|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}"
    rf"|{_MONTHS}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?,?[ \t]+\d{{4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?[ \t]+{_MONTHS}\.?,?[ \t]+\d{{4}}"
    rf"|\d{{1,2}}[ \t\n/.\-A-Z]+{_MONTHS}(?:[ \t\n/.\-A-Z]+{_MONTHS})?[ \t\n/.\-A-Z]+\d{{2,4}})"
)
_DOB_LABELS = r"d\.?o\.?b\.?|date[ \t\n]+of[ \t\n]+birth|date[ \t\n]+de[ \t\n]+naissance|fecha[ \t\n]+de[ \t\n]+nacimiento|geburtstag|birth[ \t\n]*date|birthday|born(?:[ \t\n]+on)?|تولد|تاریخ\s+تولد"
DOB_RE = re.compile(
    rf"(?i)\b(?:{_DOB_LABELS})\b[A-Za-z \t\n/.\-é:]{{0,30}}({DATE_PATTERN})"
)

# Standalone calendar-date recognizers. Passport / ID cards print birth, issue
# and expiry dates in these shapes; they are recognised even without a label
# (a "Date of birth" label only raises the entity to DATE_OF_BIRTH).
_DATE_DAY_MONTH_YEAR_RE = re.compile(
    rf"(?i)(?<![\w])\d{{1,2}}(?:st|nd|rd|th)?[ \t]+{_MONTHS}\.?[ \t]*[,'\-]?[ \t]*\d{{4}}(?![\d])"
)
_DATE_MONTH_DAY_YEAR_RE = re.compile(
    rf"(?i)(?<![\w]){_MONTHS}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?[ \t]*,[ \t]*\d{{4}}(?![\d])"
)
_DATE_NUMERIC_RE = re.compile(
    r"(?<![\d/.\-])(?:\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})(?![\d/.\-])"
)

# Bilingual passport dates: Canadian / international passports print the birth,
# issue and expiry dates with both the English and French month abbreviations on
# one line, separated by a slash (often with no space after it), e.g.
#     "01 JAN /JAN 85"    "14 JAN /JAN 13"    "05 FEV /FEB 24"
# Only real month words are accepted on either side of the slash, so ordinary
# "10 USD /EUR 20"-style text can never match.
_BILINGUAL_MONTH_WORD = (
    r"(?:janv(?:ier)?|jan(?:uary)?|f[ée]vr(?:ier)?|f[ée]v|feb(?:ruary)?|"
    r"mars|mar(?:ch)?|avr(?:il)?|apr(?:il)?|mai|may|juin|jun(?:e)?|"
    r"juil(?:let)?|jul(?:y)?|ao[uû]t?|aug(?:ust)?|"
    r"sept(?:embre)?|sep(?:t(?:ember)?)?|oct(?:obre|ober)?|"
    r"nov(?:embre|ember)?|d[ée]c(?:embre|ember)?)"
)
_DATE_BILINGUAL_RE = re.compile(
    rf"(?i)(?<![\w/])(?:0?[1-9]|[12]\d|3[01])[ \t]+{_BILINGUAL_MONTH_WORD}"
    rf"[ \t]*[/Il|1][ \t]*{_BILINGUAL_MONTH_WORD}[ \t]+(?:\d{{4}}|\d{{2}})(?![\w/])"
)

# Iranian specific patterns
IRANIAN_NATIONAL_ID_RE = re.compile(r"(?<![\d\-])(\d{10})(?![\d\-])")
IRANIAN_PHONE_RE = re.compile(r"(?<![\w+])(?:\+98|0)?9\d{9}(?![\w\-])")
JALALI_DATE_RE = re.compile(r"\b1[34]\d{2}[/.\-][01]?\d[/.\-][0-3]?\d\b")

# A value is one full date when it matches any of these end-to-end.
_FULL_DATE_RES = (
    _DATE_DAY_MONTH_YEAR_RE,
    _DATE_MONTH_DAY_YEAR_RE,
    _DATE_BILINGUAL_RE,
    _DATE_NUMERIC_RE,
    JALALI_DATE_RE,
)

_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|"
    r"Way|Place|Pl|Parkway|Pkwy|Highway|Hwy|Terrace|Ter|Square|Sq|Trail|Trl|Plaza|Plz|Alley|"
    r"Crescent|Close|Grove|Row|Mews)"
)
STREET_ADDRESS_RE = re.compile(
    rf"\b\d{{1,6}}[A-Za-z]?[ \t]+(?:[NSEW]\.?[ \t]+)?(?:[A-Z0-9][\w'.\-]*[ \t]+){{0,4}}?{_STREET_SUFFIX}\b\.?"
    r"(?:,?[ \t]*(?:Apt|Apartment|Suite|Ste|Unit|Fl|Floor|#)\.?[ \t]*#?[\w\-]+)?"
)
_US_STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|"
    "NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY"
)
CITY_STATE_ZIP_RE = re.compile(
    rf"\b[A-Z][A-Za-z.'\-]+(?:[ \t]+[A-Z][A-Za-z.'\-]+){{0,3}},[ \t]*(?:{_US_STATES})[ \t]+\d{{5}}(?:-\d{{4}})?\b"
)
PO_BOX_RE = re.compile(r"(?i)\bP\.?[ \t]?O\.?[ \t]+Box[ \t]+\d{1,6}\b")
UK_POSTCODE_RE = re.compile(r"\b(?:[A-PR-UWYZ][A-HK-Y]?\d[A-Z\d]?)[ \t]+\d[ABD-HJLNP-UW-Z]{2}\b")

SECRET_PATTERNS = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                       # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),             # GitHub token
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),          # Slack token
    re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),                 # Google API key
    re.compile(r"\b[sr]k_(?:live|test)_[0-9A-Za-z]{16,}\b"),   # Stripe key
    re.compile(r"\beyJ[\w\-]{8,}\.[\w\-]{8,}\.[\w\-]{8,}\b"),  # JWT
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),                 # OpenAI-style key
]
HIGH_ENTROPY_TOKEN_RE = re.compile(r"(?<![\w])[A-Za-z0-9+/_\-]{24,}={0,2}(?![\w])")

# Labels whose *value* is sensitive. Value must follow a ":" "=" "#" or "No." etc.
_SECRET_LABELS = (
    r"password|passwd|pwd|passcode|pass[ \t]?code|pin(?:[ \t]?code)?|api[ \t_\-]?key|secret(?:[ \t_\-]?key)?|"
    r"access[ \t_\-]?(?:key|token)|auth(?:entication)?[ \t_\-]?token|token|username|user[ \t_\-]?(?:name|id)|login"
)
_ID_LABELS = (
    r"passport|passeport|passport\s+no\.?/\s*n°?\s*de\s*passeport|document|"
    r"driver'?s?[ \t]+licen[cs]e|licen[cs]e|account|acct|a/c|routing|sort[ \t]+code|swift|bic|"
    r"member(?:ship)?|policy|patient|mrn|medical[ \t]+record|employee|staff|student|customer|client|"
    r"tax(?:payer)?|tin|ein|itin|vat|nhs|national[ \t]+(?:insurance|id)|ni|id|identification|"
    r"case|claim|card|badge|voter|visa|permit"
)
# NOTE: bare document-reference labels (invoice, order, reference, ref, tracking,
# serial) are deliberately *not* in the list: "Ref: 88294" identifies a document,
# not a person, and flagging it was a visible false positive.
# Same-line "label: value" name forms only. The bare slash form
# ("Surname / Nom / Apellidos") is a *header row*: its value is printed on the
# next line and is handled by the cross-line pairing below - it must never
# match a name on the header line itself.
_NAME_LABELS = (
    r"surname\s*/\s*nom|given\s+names\s*/\s*pr[eé]noms|nom\s*/\s*surname|"
    r"prenoms\s*/\s*given\s+names|surname|given\s+names|first\s+name|last\s+name|"
    r"full\s+name|name|nom|prenoms|prénoms|"
    r"نام\s+و\s+نام\s+خانوادگی|نام\s+خانوادگی|نام"
)
NAME_VALUE_RE = re.compile(
    rf"(?i)\b(?:{_NAME_LABELS})\b[ \t]*[:=][ \t]*"
    r"([A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF]+(?:[ \t]+[A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF]+)*)"
)
# Only explicit "label: value" / "label=value" forms. The prose form "password is ..."
# was dropped: "your password is required" is not a secret.
SECRET_VALUE_RE = re.compile(
    rf"(?i)\b(?:{_SECRET_LABELS})\b[ \t\n]*[:=#][ \t\n]*(\S{{3,}})"
)
ID_VALUE_RE = re.compile(
    rf"(?i)\b(?:{_ID_LABELS})\b[ \t\n]*(?:no\.?|num(?:ber)?\.?|#|id|code)?[ \t\n]*[:#=\-]?[ \t\n]*"
    r"([A-Za-z0-9][A-Za-z0-9\-/.]{3,}[A-Za-z0-9])"
)

LONG_NUMBER_RE = re.compile(r"(?<![\w])\d(?:[ \t\-./]?\d){7,}(?![\w])")
_NUMERIC_DATE_RE = re.compile(r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}")
_ALPHANUM_ID_RE = re.compile(r"(?<![\w])([A-Z]{1,3}[0-9]{5,9}|[0-9]{5,9}[A-Z]{1,3})(?![\w])")


def _find_phonenumbers(text: str) -> Iterable[PIISpan]:
    """Google libphonenumber matcher, filtered against classic OCR false positives."""
    try:
        import phonenumbers
        from phonenumbers import Leniency, PhoneNumberMatcher
    except ImportError:  # pragma: no cover - optional dependency
        return []

    spans = []
    seen = set()
    for region in ("US", "GB", "CA", "AU", "IN", "DE", "FR", "IE", "NZ", "ZA"):
        try:
            for match in PhoneNumberMatcher(text, region, leniency=Leniency.VALID):
                key = (match.start, match.end)
                if key in seen or not _plausible_phone_match(match.raw_string):
                    continue
                seen.add(key)
                spans.append(PIISpan(match.start, match.end, "PHONE", 0.9, "phonenumbers"))
        except Exception:  # pragma: no cover - library edge cases
            continue
    return spans


def _pattern_spans(text: str) -> List[PIISpan]:
    spans: List[PIISpan] = []
    numeric = _numeric_view(text)

    def add(m: re.Match, entity: str, score: float, group: int = 0, source: str = "pattern"):
        s, e = m.span(group)
        if e > s:
            spans.append(PIISpan(s, e, entity, score, source))

    for m in EMAIL_RE.finditer(text):
        add(m, "EMAIL", 0.95)
    for m in EMAIL_OCR_RE.finditer(text):
        domain = m.group().split("@", 1)[1]
        if len(domain) >= 4 and any(c.isalpha() for c in domain):
            add(m, "EMAIL", 0.8)

    for m in URL_RE.finditer(text):
        add(m, "URL", 0.6)

    # Card numbers (Luhn + IIN prefix sanity)
    for m in CREDIT_CARD_RE.finditer(numeric):
        d = _digits(m.group())
        if 13 <= len(d) <= 19 and d[0] in "23456" and luhn_valid(d):
            add(m, "CREDIT_CARD", 0.95)

    for m in IBAN_RE.finditer(text.upper()):
        if iban_valid(m.group()):
            add(m, "IBAN", 0.95)

    for m in SSN_RE.finditer(numeric):
        if ssn_valid(m.group(1), m.group(2), m.group(3)):
            add(m, "SSN", 0.85)
    for m in SSN_CONTEXT_RE.finditer(numeric):
        d = m.group(1)
        if ssn_valid(d[:3], d[3:5], d[5:]):
            add(m, "SSN", 0.9, group=1)

    for m in NANP_PHONE_RE.finditer(numeric):
        add(m, "PHONE", 0.85)
    for m in FORMATTED_PHONE_RE.finditer(numeric):
        add(m, "PHONE", 0.85)
    for m in INTL_PHONE_RE.finditer(numeric):
        if 8 <= len(_digits(m.group())) <= 15:
            add(m, "PHONE", 0.8)
    spans.extend(_find_phonenumbers(numeric))

    for m in IPV4_RE.finditer(numeric):
        try:
            ip = ipaddress.ip_address(m.group())
            if not (ip.is_unspecified or str(ip).startswith("0.")):
                add(m, "IP_ADDRESS", 0.8)
        except ValueError:
            pass
    for m in IPV6_RE.finditer(text):
        if m.group().count(":") >= 2:
            try:
                ipaddress.ip_address(m.group())
                add(m, "IP_ADDRESS", 0.8)
            except ValueError:
                pass
                
    for m in IRANIAN_NATIONAL_ID_RE.finditer(numeric):
        if iranian_national_id_valid(m.group(1)):
            add(m, "ID_NUMBER", 0.95)
            
    for m in IRANIAN_PHONE_RE.finditer(numeric):
        add(m, "PHONE", 0.9)
        
    for m in JALALI_DATE_RE.finditer(text):
        add(m, "DATE_OF_BIRTH", 0.85)

    for m in DOB_RE.finditer(text):
        add(m, "DATE_OF_BIRTH", 0.85, group=1)

    # Standalone calendar dates ("07 Aug 1999", "Aug 07, 1999", numeric forms and
    # the bilingual "01 JAN /JAN 85" passport shape).
    for pattern in (_DATE_DAY_MONTH_YEAR_RE, _DATE_MONTH_DAY_YEAR_RE, _DATE_BILINGUAL_RE, _DATE_NUMERIC_RE):
        for m in pattern.finditer(text):
            add(m, "DATE", 0.75)

    for m in STREET_ADDRESS_RE.finditer(text):
        add(m, "ADDRESS", 0.8)
    for m in CITY_STATE_ZIP_RE.finditer(text):
        add(m, "ADDRESS", 0.8)
    for m in PO_BOX_RE.finditer(text):
        add(m, "ADDRESS", 0.8)
    for m in UK_POSTCODE_RE.finditer(text):
        add(m, "ADDRESS", 0.65)

    for pattern in SECRET_PATTERNS:
        for m in pattern.finditer(text):
            add(m, "SECRET", 0.95)
    for m in HIGH_ENTROPY_TOKEN_RE.finditer(text):
        tok = m.group()
        classes = sum(bool(re.search(p, tok)) for p in (r"[a-z]", r"[A-Z]", r"\d"))
        if classes == 3 and shannon_entropy(tok) >= 4.0:
            add(m, "SECRET", 0.7)

    # Context recognizers: redact only the value after a sensitive label.
    for m in SECRET_VALUE_RE.finditer(text):
        add(m, "SECRET", 0.85, group=1, source="context")
    for m in ID_VALUE_RE.finditer(numeric):
        value = m.group(1)
        if sum(c.isdigit() for c in value) >= 3:
            add(m, "ID_NUMBER", 0.8, group=1, source="context")
            
    for m in NAME_VALUE_RE.finditer(text):
        value = m.group(1)
        if not _NUMERIC_DATE_RE.fullmatch(value) and _looks_like_person_name(value):
            add(m, "PERSON", 0.8, group=1, source="context")

    # Generic long digit runs (>= 8 digits) -> likely an identifier (plain dates excluded).
    for m in LONG_NUMBER_RE.finditer(numeric):
        if (len(_digits(m.group())) >= 8 and not _NUMERIC_DATE_RE.fullmatch(m.group())
                and not IPV4_RE.fullmatch(m.group())):
            add(m, "ID_NUMBER", 0.7)

    # Isolated alphanumeric strings that resemble ID documents (e.g., ZE000509)
    for m in _ALPHANUM_ID_RE.finditer(text):
        if not _NUMERIC_DATE_RE.fullmatch(m.group()):
            add(m, "ID_NUMBER", 0.75)

    # ----------------------------------------------------------------------- #
    # Machine Readable Zone (passport / ID cards): TD1 / TD2 / TD3 lines
    # ----------------------------------------------------------------------- #
    spans.extend(_mrz_spans(text))

    # Cross-line field pairing ("Surname / Nom / Apellidos" -> "DE LA PAZ"):
    # identity-document headers label the value printed on the next line.
    spans.extend(_crossline_label_value_spans(text))

    return spans


# --------------------------------------------------------------------------- #
# Contextual person-name recognizer (dependency free)
# --------------------------------------------------------------------------- #
# Names are only flagged when a strong *context cue* is present, which keeps
# precision high on prose: salutations ("Dear Jane Doe"), honorifics ("Mr. Smith"),
# labelled fields ("Name: Jane Doe"), sign-offs ("Sincerely,\nJane Doe") and the
# header line of a contact block (a bare "First Last" line adjacent to a phone or
# e-mail). Every name found this way is then propagated to its other occurrences.

_NAME_TOKEN = r"[A-Z][a-z]+(?:['\-][A-Z]?[a-z]+)?"
_NAME = rf"{_NAME_TOKEN}(?:[ \t]+(?:[A-Z]\.[ \t]+)?{_NAME_TOKEN}){{0,2}}"
_HONORIFIC = r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Madam)\.?"

SALUTATION_NAME_RE = re.compile(
    rf"\b(?:Dear|Attn\.?|Attention:?)[ \t]+(?:{_HONORIFIC}[ \t]+)?({_NAME})"
)
HONORIFIC_NAME_RE = re.compile(rf"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof)\.?[ \t]+({_NAME})")
LABELLED_NAME_RE = re.compile(
    r"(?i:\b(?:full[ \t]+name|first[ \t]+name|last[ \t]+name|surname|name|signed|signature|"
    r"patient|employee|applicant|cardholder|card[ \t]+holder|account[ \t]+holder|"
    r"recipient|sender|beneficiary|insured|guardian|contact))"
    rf"[ \t]*[:\-][ \t]*({_NAME})"
)
CLOSING_NAME_RE = re.compile(
    r"(?im)^[ \t]*(?:sincerely|regards|best[ \t]+regards|kind[ \t]+regards|warm[ \t]+regards|"
    r"best[ \t]+wishes|respectfully|yours[ \t]+(?:truly|sincerely|faithfully)|cordially|cheers|"
    r"thanks|thank[ \t]+you|best)[ \t]*,?[ \t]*\n[ \t]*"
    rf"(?-i:({_NAME}))[ \t]*$"
)
BARE_NAME_LINE_RE = re.compile(rf"^[ \t]*((?:{_HONORIFIC}[ \t]+)?{_NAME_TOKEN}(?:[ \t]+(?:[A-Z]\.[ \t]+)?{_NAME_TOKEN}){{1,2}})[ \t]*,?[ \t]*$")

# Capitalised words that frequently start lines but are not names.
_NON_NAME_WORDS = {
    "the", "a", "an", "and", "of", "to", "for", "in", "on", "at", "by", "with", "from", "re",
    "dear", "hi", "hello", "sincerely", "regards", "best", "thanks", "thank", "you", "yours",
    "hiring", "manager", "director", "officer", "assistant", "aide", "administrative", "executive",
    "team", "department", "dept", "human", "resources", "sales", "support", "service", "services",
    "customer", "office", "company", "corporation", "inc", "ltd", "llc", "corp", "group", "bank",
    "university", "college", "school", "hospital", "clinic", "street", "avenue", "road", "city",
    "state", "county", "country", "united", "states", "kingdom", "new", "north", "south", "east",
    "west", "subject", "date", "phone", "email", "mail", "address", "fax", "mobile", "tel",
    "website", "name", "page", "total", "invoice", "receipt", "order", "summary", "profile",
    "experience", "education", "skills", "objective", "references", "contact", "information",
    "cover", "letter", "resume", "curriculum", "vitae", "senior", "junior", "lead", "chief",
    "head", "president", "vice", "engineer", "developer", "designer", "analyst", "consultant",
    "specialist", "coordinator", "representative", "agent", "recruiter", "recruitment", "talent",
    "acquisition", "board", "committee", "council", "to whom", "whom", "it", "may", "concern",
    "january", "february", "march", "april", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "madam", "sir", "all", "everyone", "there", "colleagues", "friends", "customers",
}


def _plausible_name(surface: str) -> bool:
    tokens = [t.strip(".,") for t in re.split(r"[ \t]+", surface.strip()) if t.strip(".,")]
    tokens = [t for t in tokens if not re.fullmatch(_HONORIFIC, t + ".") and not re.fullmatch(r"[A-Z]", t)]
    if not tokens:
        return False
    return not any(t.lower() in _NON_NAME_WORDS for t in tokens)


def _strip_trailing_non_names(text: str, s: int, e: int) -> int:
    """Drops trailing capitalised non-name words, e.g. "Jane Doe Manager" -> "Jane Doe"."""
    words = list(re.finditer(r"\S+", text[s:e]))
    while len(words) > 1 and words[-1].group().strip(".,").lower() in _NON_NAME_WORDS:
        words.pop()
    return s + words[-1].end() if words else e


def _context_name_spans(text: str, anchors: List[PIISpan]) -> List[PIISpan]:
    spans: List[PIISpan] = []

    def add(m: re.Match, score: float):
        s, e = m.span(1)
        e = _strip_trailing_non_names(text, s, e)
        if e > s and _plausible_name(text[s:e]):
            spans.append(PIISpan(s, e, "PERSON", score, "context"))

    for m in SALUTATION_NAME_RE.finditer(text):
        add(m, 0.85)
    for m in HONORIFIC_NAME_RE.finditer(text):
        add(m, 0.85)
    for m in LABELLED_NAME_RE.finditer(text):
        add(m, 0.85)
    for m in CLOSING_NAME_RE.finditer(text):
        add(m, 0.85)

    # Contact block header: a bare "First Last" line within two lines of a phone / e-mail.
    contact_lines = set()
    line_starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]

    def line_of(pos: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    for a in anchors:
        if a.entity in ("PHONE", "EMAIL"):
            contact_lines.add(line_of(a.start))
    lines = text.split("\n")
    for ln in contact_lines:
        for cand in range(max(0, ln - 2), min(len(lines), ln + 3)):
            if cand == ln:
                continue
            m = BARE_NAME_LINE_RE.match(lines[cand])
            if m and _plausible_name(m.group(1)):
                s = line_starts[cand] + m.start(1)
                spans.append(PIISpan(s, s + len(m.group(1)), "PERSON", 0.75, "context"))
    return spans


def _propagate_names(text: str, spans: List[PIISpan]) -> List[PIISpan]:
    """Redacts every other occurrence of an already detected full name (coreference)."""
    names = {text[s.start:s.end].strip() for s in spans if s.entity == "PERSON"}
    extra: List[PIISpan] = []
    for name in names:
        if len(name) < 4 or " " not in name:
            continue
        for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text):
            extra.append(PIISpan(m.start(), m.end(), "PERSON", 0.8, "coreference"))
    return extra


# --------------------------------------------------------------------------- #
# Multilingual field vocabulary and cross-line label -> value pairing
# --------------------------------------------------------------------------- #
# Identity documents print a multilingual *header* on one line and the value on
# the line directly below it (the line reconstruction in ml_pipeline preserves
# that order; the label-to-value gap never exceeds one line box):
#
#     Surname / Nom / Apellidos
#     DE LA PAZ
#
# The header line itself is never PII; the value line inherits the field's type:
# surnames / given names are people, passport numbers are IDs and birth / issue
# / expiry values are dates. This is what catches the ALL-CAPS names the cased
# NER model misses ("DE LA PAZ", "MICHELLE").

# Field / country / utility words that must never be classified as a person.
_MULTILINGUAL_HEADER_WORDS = {
    "america", "amount", "anotaciones", "apellidos", "authority", "autoridad",
    "autorite", "autorité", "bill", "billing", "birth", "caducidad", "code",
    "codigo", "código", "currency", "date", "delivrance", "délivrance",
    "department", "endorsements", "expedicion", "expedición", "expiration",
    "fecha", "given", "invoice", "issue", "lieu", "lugar", "mentions", "nacimiento",
    "nacionalidad", "naissance", "names", "nationalite", "nationalité",
    "nationality", "nom", "nombres", "page", "pasaporte", "passeport",
    "passport", "place", "prenoms", "prénoms", "receipt", "see", "sex",
    "sexe", "sexo", "speciales", "spéciales", "state", "states", "subtotal",
    "tax", "tipo", "total", "type", "united", "usd", "utility",
}
_NON_NAME_WORDS |= _MULTILINGUAL_HEADER_WORDS


def _strip_combining(value: str) -> str:
    """
    Removes combining accents ("e" + U+0301 -> "e").

    OCR emits accented words precomposed ("Prénoms"), decomposed ("Pre\\u0301noms")
    or plainly ASCII; every variant must hit the same vocabulary entry.
    """
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", value or "") if not unicodedata.combining(ch)
    )


def _fold_word(value: str) -> str:
    """Case- and accent-insensitive form used for every vocabulary lookup."""
    return _strip_combining(value).lower()


_FOLDED_NON_NAME_WORDS = {_fold_word(word) for word in _NON_NAME_WORDS}

# Nobiliary / patronymic particles: allowed *inside* a name, never on their own.
_NAME_PARTICLES = {
    "de", "del", "la", "van", "von", "al", "el", "bin", "bint", "di", "da",
    "dos", "der", "den", "ter", "ten", "st", "mac", "mc",
}

# One person-name token: a Latinate word (accents included), a Persian word or
# a single initial ("J.").
_NAME_WORD_RE = re.compile(
    r"^[A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF][A-Za-zÀ-ÖØ-öø-ÿ'’\-\u0600-\u06FF]*$"
)
_WORD_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF]+")


def _is_non_name_surface(surface: str) -> bool:
    """
    True when an entity surface is field-label / country / document vocabulary
    rather than a person ("UNITED STATES OF AMERICA", "PASSPORT", "SEE PAGE").
    """
    tokens = [
        t.strip(".,:").lower()
        for t in re.split(r"[ \t]+", _strip_combining(surface).strip())
        if t.strip(".,:")
    ]
    if not tokens:
        return True
    words = [t for t in tokens if not _fold_word(t) in _NAME_PARTICLES and not re.fullmatch(r"[a-z]", t)]
    if not words:
        return True
    return all(_fold_word(w) in _FOLDED_NON_NAME_WORDS for w in words)


def _looks_like_person_name(surface: str) -> bool:
    """
    Shape test for the *value line* of a name field on an identity document.

    Accepts multi-part surnames with particles ("DE LA PAZ", "VAN DER BERG"),
    single given names ("MICHELLE"), accented French/Spanish forms, initials
    ("J. Smith") and Persian names; rejects field labels, country names and
    document words.
    """
    tokens = [
        t.strip(".,;:")
        for t in re.split(r"[ \t]+", _strip_combining(surface).strip())
        if t.strip(".,;:")
    ]
    if not tokens or len(tokens) > 5:
        return False
    substantial: List[str] = []
    for token in tokens:
        if not _NAME_WORD_RE.match(token):
            return False
        folded = _fold_word(token)
        if len(token) == 1:
            continue                     # middle initial
        if folded in _NAME_PARTICLES:
            continue
        if folded in _FOLDED_NON_NAME_WORDS:
            return False
        substantial.append(token)
    return any(len(t) >= 3 for t in substantial)


# Field-label word -> field kind. A line is a *header row* only when every one
# of its tokens is known here (plus separators / punctuation).
_FIELD_WORD_KINDS = {
    # surname / family-name field
    "surname": "SURNAME", "apellidos": "SURNAME", "last": "SURNAME",
    "familienname": "SURNAME", "nachname": "SURNAME",
    "خانوادگی": "SURNAME", "فامیل": "SURNAME",
    # given-names field
    "given": "GIVEN", "names": "GIVEN", "prenoms": "GIVEN", "prénoms": "GIVEN",
    "nombres": "GIVEN", "first": "GIVEN", "vorname": "GIVEN",
    # generic name field
    "nom": "SURNAME", "name": "NAME", "نام": "NAME",
    # document-number field
    "passport": "DOC_NO", "passeport": "DOC_NO", "pasaporte": "DOC_NO",
    "پاسپورت": "DOC_NO", "گذرنامه": "DOC_NO", "شناسنامه": "DOC_NO",
    "document": "DOC_NO", "doc": "DOC_NO", "no": "DOC_NO", "number": "DOC_NO",
    "num": "DOC_NO", "شماره": "DOC_NO", "کد": "DOC_NO",
    # dates
    "date": "DATE_FIELD", "fecha": "DATE_FIELD", "birth": "DOB", "naissance": "DOB",
    "nacimiento": "DOB", "تاریخ": "DATE_FIELD", "تولد": "DOB",
    "issue": "ISSUE", "delivrance": "ISSUE", "délivrance": "ISSUE",
    "expedicion": "ISSUE", "expedición": "ISSUE",
    "expiration": "EXPIRY", "expiry": "EXPIRY", "caducidad": "EXPIRY",
    # other printed fields (recognised as headers, no redacted value line)
    "place": "PLACE", "lieu": "PLACE", "lugar": "PLACE", "محل": "PLACE",
    "sex": "SEX", "sexe": "SEX", "sexo": "SEX", "جنسیت": "SEX",
    "nationality": "NATIONALITY", "nationalite": "NATIONALITY",
    "nationalité": "NATIONALITY", "nacionalidad": "NATIONALITY",
    "ملیت": "NATIONALITY", "تابعیت": "NATIONALITY",
    "authority": "AUTHORITY", "autorite": "AUTHORITY", "autorité": "AUTHORITY",
    "autoridad": "AUTHORITY", "issuing": "AUTHORITY",
    "endorsements": "OTHER", "mentions": "OTHER", "speciales": "OTHER",
    "spéciales": "OTHER", "anotaciones": "OTHER", "department": "OTHER",
    "type": "OTHER", "country": "OTHER", "state": "OTHER", "states": "OTHER",
    "see": "OTHER", "page": "OTHER", "code": "OTHER", "codigo": "OTHER",
    "código": "OTHER",
    # connectives / elisions allowed inside a header row
    "of": "CONN", "de": "CONN", "du": "CONN", "des": "CONN", "del": "CONN",
    "la": "CONN", "and": "CONN", "و": "CONN", "n": "CONN", "d": "CONN", "l": "CONN",
}

# Accent-insensitive view of the map above ("Prénoms" == "Prenoms" == decomposed).
_FOLDED_FIELD_WORD_KINDS = {_fold_word(word): kind for word, kind in _FIELD_WORD_KINDS.items()}

# Single-word headers only pair a value for these unambiguous labels.
_STRONG_SINGLE_LABELS = {
    "surname", "apellidos", "passport", "passeport", "pasaporte", "نام", "شماره",
}

# Kinds whose value line is actively classified, in precedence order.
_PAIRED_KINDS = ("DOC_NO", "DOB", "ISSUE", "EXPIRY", "SURNAME", "GIVEN", "NAME")


def _classify_header_line(line: str) -> Optional[str]:
    """
    Returns the field kind when ``line`` consists *only* of known field labels
    ("Surname / Nom / Apellidos", "Passport No. / No. du Passeport"), else None.
    Lines containing digits are values, never headers.
    """
    if any(ch.isdigit() for ch in line):
        return None
    tokens = [t.lower() for t in _WORD_TOKEN_RE.findall(_fold_word(line))]
    if not tokens or len(tokens) > 12:
        return None
    kinds: List[str] = []
    for token in tokens:
        kind = _FOLDED_FIELD_WORD_KINDS.get(token)
        if kind is None:
            return None
        kinds.append(kind)
    if len(tokens) < 2 and tokens[0] not in _STRONG_SINGLE_LABELS:
        return None
    for kind in _PAIRED_KINDS:
        if kind in kinds:
            return kind
    return "HEADER"


def _value_looks_like_document_number(value: str) -> bool:
    """
    The value under a "Passport No." / "Document No." header: the 8-10 digit
    passport number ("910239248") or an alphanumeric document number
    ("L898902C3").
    """
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    if not compact or len(compact) > 16:
        return False
    digits = sum(c.isdigit() for c in compact)
    return digits >= 5


def _looks_like_full_date(value: str) -> bool:
    """True when the whole (trimmed) value is one calendar date."""
    candidate = (value or "").strip().rstrip(".,;")
    if not candidate:
        return False
    return any(rx.fullmatch(candidate) for rx in _FULL_DATE_RES)


def _crossline_label_value_spans(text: str) -> List[PIISpan]:
    """
    Pairs a multilingual field header with the value printed on the next line.

    Implements the passport / ID-card layout ("header row" -> "value row"): the
    header line is skipped entirely and the line directly below it is redacted
    according to the field type -- names -> PERSON, passport numbers ->
    ID_NUMBER, birth dates -> DATE_OF_BIRTH, issue / expiry dates -> DATE.
    """
    spans: List[PIISpan] = []
    lines = text.split("\n")
    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    for index, line in enumerate(lines):
        kind = _classify_header_line(line)
        if kind is None or kind == "HEADER":
            continue
        nxt = index + 1
        while nxt < len(lines) and not lines[nxt].strip():
            nxt += 1
        if nxt >= len(lines):
            continue
        value_line = lines[nxt]
        if _classify_header_line(value_line) is not None:
            continue                     # two labels stacked: value belongs elsewhere
        value = value_line.strip()
        if not value or len(value) > 60:
            continue

        entity: Optional[str] = None
        if kind == "DOC_NO":
            if _value_looks_like_document_number(value):
                entity = "ID_NUMBER"
        elif kind == "DOB":
            if _looks_like_full_date(value):
                entity = "DATE_OF_BIRTH"
        elif kind in ("ISSUE", "EXPIRY"):
            if _looks_like_full_date(value):
                entity = "DATE"
        elif kind in ("SURNAME", "GIVEN", "NAME"):
            if _looks_like_person_name(value):
                entity = "PERSON"
        if entity is None:
            continue
        start = offsets[nxt] + (len(value_line) - len(value_line.lstrip()))
        spans.append(PIISpan(start, start + len(value), entity, 0.9, "header_pair"))
    return spans


def _header_line_ranges(text: str) -> List[Tuple[int, int]]:
    """Character ranges of every pure field-header row of ``text``."""
    ranges: List[Tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        if _classify_header_line(line) is not None:
            ranges.append((pos, pos + len(line)))
        pos += len(line) + 1
    return ranges


# --------------------------------------------------------------------------- #
# Public helpers for the OCR pipeline layer (ml_pipeline.py)
# --------------------------------------------------------------------------- #

def field_header_kind(line: str) -> Optional[str]:
    """
    Public wrapper around the header classifier: the field kind of a pure label
    row ("Surname / Nom / Apellidos" -> "SURNAME", "Date of birth" -> "DOB"),
    or None when the row is not a label-only row.
    """
    return _classify_header_line(line)


def looks_like_date_value(text: str) -> bool:
    """Public wrapper: True when the whole (trimmed) value is one calendar date."""
    return _looks_like_full_date(text)


# --------------------------------------------------------------------------- #
# Machine Readable Zone (TD1 / TD2 / TD3) detection
# --------------------------------------------------------------------------- #
# OCR output against real passport photos is noisy: it inserts spaces between
# characters ("P < D E L A P A Z < < ..."), misreads the "<" filler as runs of
# K / L / c / ( and often drops the 3-letter issuing-state code entirely
# ("P<DELAPAZ<<MICHELLE..."). The detector therefore normalises the candidate
# first and then matches the structural TD1 / TD2 / TD3 shapes; every matched
# line is redacted *in full* as entity MRZ.

_MRZ_FILLER_MISREADS = "KLC"
# Brackets and parentheses are never printed on a data page: they are always
# the "<" filler, whatever the OCR thought it saw.
_MRZ_PUNCT_FILLERS = "()[]{}"
_MRZ_CHARSET_RE = re.compile(r"^[A-Z0-9<]+$")
# TD3 / TD2 / TD1 line 1: document type + "<" + at least 5 more MRZ characters.
# Digits are allowed right after the prefix: a TD1 (ID-card) first row carries
# the document number ("I<UTOD231458907...") in the same row.
_MRZ_NAME_LINE_RE = re.compile(r"^[PIVCA][<O0][A-Z0-9<]{5,}")
# Data lines carry a sex marker followed by the 6-digit expiry date.
_MRZ_SEX_EXPIRY_RE = re.compile(r"[MF<]\d{6}")
# Issuing-state codes: the ICAO Doc 9303 specimen code ("UTO") plus the most
# common real ones, used only to strip the prefix before the surname field.
_MRZ_STATE_CODES = {
    "UTO", "USA", "GBR", "CAN", "FRA", "DEU", "ITA", "ESP", "NLD", "BEL",
    "CHE", "AUT", "SWE", "NOR", "DNK", "FIN", "IRL", "POL", "PRT", "GRC",
    "TUR", "RUS", "UKR", "CHN", "JPN", "KOR", "IND", "PAK", "IRN", "IRQ",
    "SAU", "ARE", "EGY", "ZAF", "BRA", "ARG", "MEX", "AUS", "NZL", "SGP",
    "MYS", "THA", "IDN", "VNM", "PHL",
}


def _repair_mrz_fillers(compact: str) -> str:
    """
    Restores "<" fillers that OCR misread.

    Brackets / parentheses never occur on a data page and are always taken for
    the filler; K / L / C are only repaired when they form a run of two or more
    identical characters ("KKK" -> "<<<", the shape OCR produces for a filler
    run) or sit between two fillers - so a genuine name letter is never
    rewritten ("ERIKSSON", "JACK<<").
    """
    chars = ["<" if ch in _MRZ_PUNCT_FILLERS else ch for ch in compact]
    out = list(chars)
    index, length = 0, len(chars)
    while index < length:
        char = chars[index]
        if char in _MRZ_FILLER_MISREADS:
            end = index
            while end < length and chars[end] == char:
                end += 1
            if end - index >= 2:
                for position in range(index, end):
                    out[position] = "<"
            elif (0 < index and index + 1 < length
                    and out[index - 1] == "<" and out[index + 1] == "<"):
                out[index] = "<"
            index = end
        else:
            index += 1
    return "".join(out)


def _mrz_line_kind(compact: str, repaired: str) -> str:
    """
    Classifies one whitespace-free candidate line: "name" for the TD1/TD2/TD3
    name line, "data" for the number lines, "" when it is not an MRZ line.
    """
    if len(compact) < 28 or not _MRZ_CHARSET_RE.match(repaired):
        return ""
    if _MRZ_NAME_LINE_RE.match(repaired):
        return "name"
    # TD1 name row (third line) carries no document-type prefix: it is pure
    # A-Z fillers. "<<" only ever appears in an MRZ, so this is unambiguous.
    if "<<" in repaired and re.fullmatch(r"[A-Z<]+", repaired):
        return "name"
    digits = sum(char.isdigit() for char in repaired)
    if (digits >= 10
            and (_MRZ_SEX_EXPIRY_RE.search(repaired) or re.search(r"[A-Z]{3}", repaired))
            and ("<" in repaired or digits >= 20)):
        return "data"
    return ""


def _mrz_name_fields(repaired: str, state_codes) -> List[str]:
    """
    Extracts the surname / given-name fields of a TD1/TD2/TD3 name line.

    Handles the missing issuing-state code ("P<DELAPAZ<<MICHELLE"): the
    3-letter prefix is stripped only when it matches a code seen on the
    companion data line or one of the well-known specimen codes, otherwise it
    is kept as part of the surname. The TD1 name row has no document-type
    prefix at all and is used as-is.
    """
    if _MRZ_NAME_LINE_RE.match(repaired):
        rest = repaired[2:]                  # drop the document-type prefix ("P<")
        if (len(rest) > 4 and rest[3].isalpha()
                and (rest[:3] in state_codes or rest[:3] in _MRZ_STATE_CODES)):
            rest = rest[3:]                  # drop the issuing-state code
    else:
        rest = repaired                      # TD1: the name row carries no prefix
    parts = [part for part in re.split(r"<{2,}", rest) if part]
    fields: List[str] = []
    for part in parts[:2]:                   # surname, then given names
        field = part.replace("<", " ").strip()
        if len(field) >= 3 and re.fullmatch(r"[A-Z'\- ]+", field):
            fields.append(field)
    return fields


def _mrz_document_number(part: str) -> str:
    """Document number candidate: printable MRZ characters, >= 5 digits, and
    never a segment that is really the issuing-state code."""
    doc = part.replace("<", "").strip()
    if len(doc) < 5 or sum(c.isdigit() for c in doc) < 5:
        return ""
    if any(code in doc for code in _MRZ_STATE_CODES):
        return ""
    return doc


def _mrz_spans(text: str) -> List[PIISpan]:
    """
    Detects Machine Readable Zone lines and returns one span per line (the
    whole line is redacted as entity MRZ) plus coreference spans that propagate
    the names and document numbers it exposes to their other occurrences on the
    page (the printed "DE LA PAZ" of the MRZ's "DELAPAZ", ...).
    """
    spans: List[PIISpan] = []
    candidates: List[Tuple[int, int, str, str]] = []   # start, end, repaired, kind
    pos = 0
    for line in text.split("\n"):
        line_end = pos + len(line)
        compact = re.sub(r"\s+", "", line.upper())
        if len(compact) >= 28:
            repaired = _repair_mrz_fillers(compact)
            kind = _mrz_line_kind(compact, repaired)
            if kind:
                candidates.append((pos, line_end, repaired, kind))
        pos = line_end + 1
    if not candidates:
        return spans

    # The entire machine-readable line is sensitive: one MRZ span per line.
    for start, end, _repaired, _kind in candidates:
        spans.append(PIISpan(start, end, "MRZ", 0.95, "mrz"))

    # The issuing-state code of a TD3/TD2 data line sits in columns 11-13.
    state_codes = set()
    for _start, _end, repaired, kind in candidates:
        if kind == "data" and len(repaired) >= 13:
            code = repaired[10:13]
            if code.isalpha():
                state_codes.add(code)

    extracted: dict = {}
    for _start, _end, repaired, kind in candidates:
        if kind == "name":
            for field in _mrz_name_fields(repaired, state_codes):
                extracted[field] = "PERSON"
        else:
            for part in (repaired[:9], repaired[5:14] if len(repaired) >= 14 else ""):
                doc = _mrz_document_number(part)
                if doc:
                    extracted[doc] = "ID_NUMBER"

    for token, entity in extracted.items():
        token = token.strip()
        if len(token) < 3:
            continue
        matches = list(re.finditer(rf"(?<!\w){re.escape(token)}(?!\w)", text, re.IGNORECASE))
        if not matches and entity == "PERSON" and " " not in token and len(token) >= 5:
            # The MRZ drops the spaces inside names ("DELAPAZ"); the printed
            # area keeps them ("DE LA PAZ"): match the letters, ignore spaces
            # and tabs - but never cross a line break.
            spaced = r"[ \t]*".join(re.escape(ch) for ch in token)
            matches = list(re.finditer(rf"(?<!\w){spaced}(?!\w)", text, re.IGNORECASE))
        for match in matches:
            spans.append(PIISpan(match.start(), match.end(), entity, 0.95, "mrz_coref"))
    return spans


# --------------------------------------------------------------------------- #
# NER recognizer (PERSON / LOCATION)
# --------------------------------------------------------------------------- #

NER_MODEL_NAME = "dslim/bert-base-NER"

# Tokens that NER models occasionally tag as PER/LOC in letters/forms but never are.
_NER_STOPWORDS = {
    "dear", "sincerely", "regards", "best", "thanks", "thank", "hi", "hello", "mr", "mrs", "ms",
    "dr", "sir", "madam", "manager", "hiring", "team", "subject", "re", "to", "from", "cc",
    "name", "date", "address", "phone", "email", "signature",
}

# Words of three or more capitals ("DE", "LA PAZ", "MICHELLE") mark a chunk
# whose all-caps text is worth re-running through the model in title case.
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")


class NERRecognizer:
    """Lazy, thread-safe wrapper around a HuggingFace token-classification pipeline."""

    def __init__(self, model_name: str = NER_MODEL_NAME, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._pipe = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._pipe is not None or self._failed:
            return self._pipe
        with self._lock:
            if self._pipe is not None or self._failed:
                return self._pipe
            try:
                from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

                tok = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
                model = AutoModelForTokenClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)
                model.eval()
                self._pipe = pipeline(
                    "token-classification",
                    model=model,
                    tokenizer=tok,
                    aggregation_strategy="first",
                    device=-1,
                )
            except Exception as exc:  # network / missing package
                logger.warning("NER model unavailable (%s); person/location names will not be detected.", exc)
                self._failed = True
        return self._pipe

    @staticmethod
    def _chunks(text: str, max_chars: int = 1500):
        """Splits text on line boundaries into chunks that fit the 512-token window."""
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                nl = text.rfind("\n", start, end)
                if nl > start:
                    end = nl + 1
            yield start, text[start:end]
            start = end

    def find(self, text: str, min_score: float = 0.80) -> List[PIISpan]:
        pipe = self._load()
        if pipe is None or not text.strip():
            return []
        spans: List[PIISpan] = []
        seen = set()
        for offset, chunk in self._chunks(text):
            if not chunk.strip():
                continue
            # All-caps text ("DE LA PAZ", "MICHELLE") is under-recognised by the
            # cased model; the same text in title case is what the model saw in
            # training, so it is evaluated as well. str.title() rewrites ASCII
            # letters one-to-one, so offsets map back exactly (isascii guard).
            variants = [chunk]
            if chunk.isascii() and _CAPS_WORD_RE.search(chunk):
                variants.append(chunk.title())
            for variant in variants:
                try:
                    results = pipe(variant)
                except Exception as exc:  # pragma: no cover
                    logger.warning("NER inference failed: %s", exc)
                    continue
                for r in results:
                    group = r.get("entity_group", "")
                    score = float(r.get("score", 0.0))
                    # Only keep PERSON; ignore LOCATION (e.g. USA, Canada) to prevent false positives
                    if group != "PER":
                        continue
                    s, e = int(r["start"]) + offset, int(r["end"]) + offset
                    if (s, e) in seen:
                        continue
                    surface = text[s:e].strip()
                    if len(surface) < 2 or not any(c.isalpha() for c in surface):
                        continue
                    if surface.lower().strip(".,:") in _NER_STOPWORDS:
                        continue
                    # Field labels / country names / document words are never names.
                    if _is_non_name_surface(surface):
                        continue
                    # Names are capitalised in typed documents; this rejects OCR junk.
                    if score < min_score or not surface[0].isupper():
                        continue
                    seen.add((s, e))
                    spans.append(PIISpan(s, e, "PERSON", round(score, 2), "ner"))
        return spans


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _merge(spans: List[PIISpan], text: str) -> List[PIISpan]:
    """Merges overlapping spans; the higher-priority entity wins the label."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s.start, -s.end))
    merged: List[PIISpan] = [spans[0]]
    for sp in spans[1:]:
        cur = merged[-1]
        if sp.start < cur.end:  # overlap
            winner = cur if (_priority(cur.entity), cur.score) >= (_priority(sp.entity), sp.score) else sp
            merged[-1] = PIISpan(
                cur.start, max(cur.end, sp.end), winner.entity, max(cur.score, sp.score), winner.source
            )
        else:
            merged.append(sp)

    # Trim whitespace / trailing punctuation from span edges and drop any LOCATION entities.
    cleaned = []
    for sp in merged:
        if sp.entity == "LOCATION":
            continue
        s, e = sp.start, sp.end
        while s < e and text[s] in " \t\n,;:":
            s += 1
        while e > s and text[e - 1] in " \t\n,;:.":
            e -= 1
        if e > s:
            cleaned.append(PIISpan(s, e, sp.entity, sp.score, sp.source))
    return cleaned


def find_pii(
    text: str,
    ner: Optional[NERRecognizer] = None,
    entities: Optional[set] = None,
) -> List[PIISpan]:
    """
    Detects PII in ``text`` and returns merged, non-overlapping character spans.

    Args:
        text: Document text (lines separated by "\\n").
        ner: Optional NER recognizer for PERSON / LOCATION entities.
        entities: Optional whitelist of entity names to keep.
    """
    if not text:
        return []
    text = text.translate(_DIGIT_TRANSLATION)
    spans = _pattern_spans(text)
    spans.extend(_context_name_spans(text, spans))
    if ner is not None:
        spans.extend(ner.find(text))
    spans.extend(_propagate_names(text, spans))
    # Field-header rows are labels, never PII: "Surname / Nom / Apellidos" must
    # not produce a PERSON span on "Nom / Apellidos" (or on any other header
    # word a looser recognizer might match).
    header_ranges = _header_line_ranges(text)
    if header_ranges:
        spans = [
            s for s in spans
            if not any(start <= s.start and s.end <= end for start, end in header_ranges)
        ]
    if entities is not None:
        spans = [s for s in spans if s.entity in entities]
    return _merge(spans, text)
