import re
import regex as reg
import unicodedata as ud
import torch
from .constants import *
from .torch_utils import default_device

# This module contains various text processing utilities, Unicode handling
# functions, and noise filters used by the main tokenizer.

# --- CJK Detection and Merge ---

# Pre-compile the regex for checking CJK scripts (Han, Hiragana, Katakana).
CJK_SC = reg.compile(r'\p{Script=Han}|\p{Script=Hiragana}|\p{Script=Katakana}')
def all_cjk(s):
   """Checks if a string consists entirely of CJK characters (fast version)."""
   return all(CJK_SC.fullmatch(c) for c in s)

def in_cjk(ch):
    """Checks if a single character is within the CJK Unicode ranges."""
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF) or \
           (0x3040 <= o <= 0x309F) or \
           (0x30A0 <= o <= 0x30FF)

def merge_cjk_runs(pieces, max_len=8):
    """Merges consecutive, short CJK character tokens into single tokens.

    This is a post-processing step to correct cases where the tokenizer might
    split a multi-character CJK word into individual characters.
    Example: `['日', '本']` -> `['日本']`.

    Args:
        pieces (list[str]): The list of tokens to process.
        max_len (int): The maximum number of single characters to merge into one token.

    Returns:
        list[str]: The processed list of tokens with CJK runs merged.
    """
    out, buf = [], []
    def flush():
        if buf:
            out.append("".join(buf)); buf.clear()
    for p in pieces:
        # If the piece is a short CJK token, add it to the buffer.
        if p and all_cjk(p) and len(p) <= 2:
            buf.append(p)
            if len(buf) >= max_len:
                flush()
        # If it's not, flush the buffer and append the current piece.
        else:
            flush()
            out.append(p)
    flush() # Final flush to catch any remaining characters in the buffer.
    return out

def in_tamil(ch): o = ord(ch); return 0x0B80 <= o <= 0x0BFF
def all_tamil(s): return bool(s) and all(in_tamil(c) for c in s)

def grapheme_clusters(s: str): return reg.findall(r'\X', s)
def count_graphemes(s: str) -> int:
    """Counts the number of user-perceived characters (grapheme clusters)."""
    return len(reg.findall(r'\X', s))

# --- Junk / Noise filters ---

def looks_like_redirect(text: str) -> bool:
    """Detects if a text is likely a Wikipedia-style redirect page."""
    if REDIRECT_TOKEN_RE.search(text):
        return True
    # As a fallback, check for the word in all caps with spaces removed.
    up = text.upper()
    letters_only = re.sub(r"\s+", "", up)  # collapse spaces between letters
    return ("REDIRECT" in letters_only) or ("WEITERLEITUNG" in letters_only)

def is_all_punct_or_space(s: str) -> bool:
    """Checks if a string consists exclusively of punctuation and/or whitespace."""
    t = s.strip()
    return t != "" and all((ch in PUNCT_SET) or ch.isspace() for ch in t)

def _is_mixed_script(tok: str) -> bool:
    """Checks if a token contains characters from multiple scripts (e.g., Latin and Cyrillic)."""
    S=set()
    for ch in tok:
        if ch.isalpha():
            name = ud.name(ch, "")
            if "CYRILLIC" in name: S.add("Cyrl")
            elif "ARABIC" in name: S.add("Arab")
            elif "GREEK" in name: S.add("Grek")
            elif ("HIRAGANA" in name) or ("KATAKANA" in name) or ("CJK" in name): S.add("Jpn")
            else: S.add("Latin")
    return len(S) >= 2

def _too_many_internal_spaces(t: str, k: int = 2) -> bool:
    """Checks if a token contains more than `k` internal space characters."""
    return t.strip().count(' ') > k

def clean_junk_runs(pieces):
    """Removes repetitive runs of junk/formatting tokens from a list."""
    cleaned = []
    for p in pieces:
        # Filter out repeating markdown-like tokens.
        if p in {"**", "***", "''", "'''", "``", "```", "==", "===", ": :"}:
            if cleaned and cleaned[-1] == p:
                continue
        # Filter out repeating punctuation tokens.
        if is_all_punct_or_space(p):
            if cleaned and is_all_punct_or_space(cleaned[-1]):
                continue
        cleaned.append(p)
    return cleaned

# --- Grapheme Handling and Protected Spans ---

def is_mark(ch: str) -> bool:
    """Checks if a character is a Unicode combining mark (e.g., an accent)."""
    return ud.category(ch).startswith('M')

def script_guess(tok: str) -> str:
    """Performs a fast, simple guess of the script/language of a token."""
    for ch in tok:
        if ch.isalpha():
            name = ud.name(ch, "")
            if "CYRILLIC" in name: return "ru"
            if "GREEK"   in name:  return "el"
            if "ARABIC"  in name:  return "ar"
            if "HEBREW"  in name:  return "he"
            if "HIRAGANA" in name or "KATAKANA" in name or "CJK" in name: return "ja"
            return "en" # Default to English/Latin script.
    return "other"

def default_allowed_boundaries(text: str, device: torch.device | None = None):
    """Determines valid splitting points in a string, respecting Unicode graphemes.

    This prevents the tokenizer from splitting in the middle of a multi-character
    grapheme, such as an emoji with a skin-tone modifier or a letter with a
    combining accent.

    Returns:
        torch.Tensor: Boolean mask where `B[k]` is True if splitting *before*
                      the k-th character is allowed.
    """
    T = len(text)
    B = torch.ones(T + 1, dtype=torch.bool, device=device or default_device())
    for k in range(1, T):
        # Don't split around a Zero-Width Joiner (used in complex emojis).
        if text[k-1] == ZWJ or text[k] == ZWJ:
            B[k] = False
        # Don't split a character from its combining mark or variation selector.
        if is_mark(text[k]) or text[k] == VS16 or text[k-1] == VS16:
            B[k] = False
    return B

def find_protected_spans(text: str):
    """Finds and merges all spans that should be treated as atomic units (URLs, emails, numbers)."""
    spans = []
    for m in URL_RE.finditer(text):   spans.append((m.start(), m.end()))
    for m in EMAIL_RE.finditer(text): spans.append((m.start(), m.end()))
    for m in NUM_RE.finditer(text):   spans.append((m.start(), m.end()))
    spans.sort()
    # Merge overlapping or adjacent spans.
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

def span_overlaps_protected(i, j, protected):
    """Checks if a span (i, j) improperly overlaps with any protected span."""
    for (s, e) in protected:
        # Check for any overlap.
        if i < e and j > s:
            # An exact match is allowed, as it means the span *is* the protected token.
            # Any other overlap is illegal.
            if not (i == s and j == e):  # exact match allowed
                return True
    return False

class ParagraphInfo:
    """A data class to hold pre-processed information about a paragraph."""
    def __init__(self, text: str, lang: str = None):
        """
        Args:
            text (str): The raw text of the paragraph.
            lang (str, optional): The language code of the text. Guessed if not provided.
        """
        self.text = text
        self.lang = lang or script_guess(text) # could use fasttext alternative
        # Pre-compute valid boundaries and protected spans for efficiency.
        self.boundary_ok = default_allowed_boundaries(text)
        self.protected = find_protected_spans(text)

    def is_legal_span(self, i: int, j: int) -> bool:
        """Checks if a span from index `i` to `j` is a valid candidate for a token.

        This method is called frequently by the DP decoder to validate potential tokens.
        """
        if not (0 <= i < j <= len(self.text)): return False
        # The start and end points must be valid Unicode boundaries.
        if not (self.boundary_ok[i].item() and self.boundary_ok[j].item()): return False
        # The span must not improperly overlap with a protected region.
        if span_overlaps_protected(i, j, self.protected):     return False
        return True
