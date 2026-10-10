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
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

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
ENTITY_INFO = {
    "CREDIT_CARD": ("Credit card", 100),
    "IBAN": ("IBAN", 100),
    "SSN": ("SSN", 100),
    "SECRET": ("Secret / key", 95),
    "EMAIL": ("Email", 90),
    "PHONE": ("Phone", 85),
    "ID_NUMBER": ("ID number", 80),
    "DATE_OF_BIRTH": ("Date of birth", 75),
    "IP_ADDRESS": ("IP address", 70),
    "ADDRESS": ("Address", 65),
    "URL": ("URL", 60),
    "PERSON": ("Person name", 55),
    "LOCATION": ("Location", 50),
}


def entity_label(entity: str) -> str:
    return ENTITY_INFO.get(entity, (entity.title(), 0))[0]


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
_DOB_LABELS = r"d\.?o\.?b\.?|date[ \t\n]+of[ \t\n]+birth|date[ \t\n]+de[ \t\n]+naissance|geburtstag|birth[ \t\n]*date|birthday|born(?:[ \t\n]+on)?|تولد|تاریخ\s+تولد"
DOB_RE = re.compile(
    rf"(?i)\b(?:{_DOB_LABELS})\b[A-Za-z \t\n/.\-é]{{0,30}}({DATE_PATTERN})"
)

# Iranian specific patterns
IRANIAN_NATIONAL_ID_RE = re.compile(r"(?<![\d\-])(\d{10})(?![\d\-])")
IRANIAN_PHONE_RE = re.compile(r"(?<![\w+])(?:\+98|0)?9\d{9}(?![\w\-])")
JALALI_DATE_RE = re.compile(r"\b1[34]\d{2}[/.\-][01]?\d[/.\-][0-3]?\d\b")

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
    r"passport|passeport|passport\s+no\.?/\s*n°?\s*de\s*passeport|driver'?s?[ \t]+licen[cs]e|licen[cs]e|account|acct|a/c|routing|sort[ \t]+code|swift|bic|"
    r"member(?:ship)?|policy|patient|mrn|medical[ \t]+record|employee|staff|student|customer|client|"
    r"tax(?:payer)?|tin|ein|itin|vat|nhs|national[ \t]+(?:insurance|id)|ni|id|identification|"
    r"case|claim|invoice|order|reference|ref|tracking|serial|card|badge|voter|visa|permit"
)
_NAME_LABELS = (
    r"surname\s*/\s*nom|given\s+names\s*/\s*pr[eé]noms|nom\s*/\s*surname|prenoms\s*/\s*given\s+names|surname|given\s+names|first\s+name|last\s+name|full\s+name|name|nom|prenoms"
)
NAME_VALUE_RE = re.compile(
    rf"(?i)\b(?:{_NAME_LABELS})\b[ \t\n/:]*([A-Z][A-Za-z\-]+(?:[ \t]+[A-Z][A-Za-z\-]+)*)"
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
                if key in seen:
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
        if not _NUMERIC_DATE_RE.fullmatch(m.group(1)):
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
    # Machine Readable Zone (MRZ) parser for Passports / IDs
    # ----------------------------------------------------------------------- #
    MRZ_RE = re.compile(r"(?<![A-Z0-9<])[A-Z0-9< ]{26,}(?![A-Z0-9<])")
    mrz_matches = [m for m in MRZ_RE.finditer(text.upper()) if "<" in m.group() and sum(c.isalpha() for c in m.group()) >= 2]
    
    mrz_texts = [m.group() for m in mrz_matches]
    for m in mrz_matches:
        add(m, "ID_NUMBER", 0.95)
        
    mrz_extracted = set()
    
    def _clean_mrz_part(part: str):
        tokens = [t.strip() for t in re.split(r'[<]+', part)]
        for t in tokens:
            t_clean = re.sub(r'^K{2,}|K{2,}$', '', t)
            if len(t_clean) >= 3:
                mrz_extracted.add(t_clean)
                print(f"DEBUG: extracted MRZ part: {t_clean}")

    if len(mrz_texts) == 2 and (len(mrz_texts[0]) >= 38 or mrz_texts[0].startswith('P')):
        # Passport (ID3)
        parts = mrz_texts[0][5:].split("<<")
        if len(parts) >= 2:
            _clean_mrz_part(parts[0])
            _clean_mrz_part(parts[1])
        doc_num = mrz_texts[1][:9].replace("<", "")
        mrz_extracted.add(doc_num)
    elif len(mrz_texts) >= 2:
        # ID1 / ID2
        doc_num = mrz_texts[0][5:14].replace("<", "")
        mrz_extracted.add(doc_num)
        name_line = mrz_texts[-1].split("<<")
        if len(name_line) >= 2:
            _clean_mrz_part(name_line[0])
            _clean_mrz_part(name_line[1])

    print(f"DEBUG: mrz_texts: {mrz_texts}")
    print(f"DEBUG: mrz_extracted: {mrz_extracted}")
    for val in mrz_extracted:
        if len(val) >= 3:
            for m in re.finditer(rf"(?i)(?<![\w]){re.escape(val)}(?![\w])", text):
                entity = "PERSON" if sum(c.isalpha() for c in val) > sum(c.isdigit() for c in val) else "ID_NUMBER"
                add(m, entity, 0.95, source="mrz_coref")

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
        for m in re.finditer(rf"(?<![\w]){re.escape(name)}(?![\w])", text):
            extra.append(PIISpan(m.start(), m.end(), "PERSON", 0.8, "coreference"))
    return extra


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
        for offset, chunk in self._chunks(text):
            if not chunk.strip():
                continue
            try:
                results = pipe(chunk)
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
                surface = text[s:e].strip()
                if len(surface) < 2 or not any(c.isalpha() for c in surface):
                    continue
                if surface.lower().strip(".,:") in _NER_STOPWORDS:
                    continue
                # Names are capitalised in typed documents; this rejects OCR junk.
                if score < min_score or not surface[0].isupper():
                    continue
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
    if entities is not None:
        spans = [s for s in spans if s.entity in entities]
    return _merge(spans, text)
