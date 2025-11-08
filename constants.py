import re
import string
import unicodedata as ud

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

EPS = 1e-12
ZWJ = "\u200D"
VS16 = "\uFE0F"

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.UNICODE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.UNICODE)
NUM_RE = re.compile(
    r"""
(?:
  [+\-]?
  (?:
    (?:\d{1,3}(?:[ ,.\u00A0]\d{3})+|\d+)
    (?:[.,]\d+)?
  )
  |
  \d{4}-\d{2}-\d{2}
)
""",
    re.UNICODE | re.VERBOSE,
)
EMOJI_RE = re.compile(
    r"["
    r"\U0001F300-\U0001FAFF"
    r"\U00002700-\U000027BF"
    r"\U00002600-\U000026FF"
    r"]",
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Redirect detection
# ---------------------------------------------------------------------------


def _spaced_letters(word: str) -> str:
    return r"\s*".join(list(word))


REDIRECT_WORDS_ASCII = ["redirect", "redirection", "weiterleitung", "yönlendirme", "yönlendir"]
REDIRECT_SPACED_ASCII = [_spaced_letters(w) for w in ["REDIRECT", "WEITERLEITUNG"]]
REDIRECT_WORDS_OTHER = [
    "перенаправ",
    "إعادةتوجيه",
    "転送",
    "umleiten",
]

REDIRECT_TOKEN_RE = re.compile(
    r"(?i)("
    + r"|".join(REDIRECT_WORDS_ASCII + REDIRECT_WORDS_OTHER + REDIRECT_SPACED_ASCII)
    + r")"
)

# ---------------------------------------------------------------------------
# Miscellaneous filters
# ---------------------------------------------------------------------------

WIKI_NOISE_RE = re.compile(r"^(\*{2,}|'{2,}|`{2,}|={2,}|[:;,#\-\u3000\s]+)$")
QUOTE_RUN_EDGE_RE = re.compile(r"(\s''+|''+\s|^''+|''+$)")

PUNCT_SET = set(string.punctuation) | {
    "«",
    "»",
    "„",
    "“",
    "”",
    "—",
    "–",
    "…",
    "·",
    "：",
    "；",
    "！",
    "？",
    "。",  # Japanese/Chinese full stop
    "、",  # Japanese/Chinese comma
    "「",
    "」",
    "『",
    "』",
}

# ---------------------------------------------------------------------------
# Affix inventories
# ---------------------------------------------------------------------------

AFFIXES = {
    "en": {
        "pre": ["re", "un", "in", "im", "dis", "mis", "non", "over", "under", "pre", "post", "anti"],
        "suf": ["ing", "ed", "er", "ers", "ly", "ness", "ment", "tion", "sion", "ous", "able", "less", "ful", "hood", "ity", "ive"],
    },
    "de": {
        "pre": ["be", "ge", "ent", "ver", "zer", "er", "ur", "miss"],
        "suf": ["ung", "en", "er", "heit", "keit", "isch", "lich", "erei", "lein", "chen", "schaft"],
    },
    "tr": {
        "pre": [],
        "suf": [
            "ler", "lar",
            "lık", "lik", "luk", "lük",
            "cı", "ci", "cu", "cü",
            "sız", "siz", "suz", "süz",
            "dan", "den", "tan", "ten",
            "yor", "ıyor", "iyor", "uyor", "üyor",
            "dı", "di", "du", "dü", "tı", "ti", "tu", "tü",
            "miş", "mış", "müş", "muş",
        ],
    },
    "ru": {
        "pre": ["по", "про", "не", "без", "раз", "под", "пред", "над", "сверх", "вы", "за", "со", "при"],
        "suf": ["ник", "чик", "щик", "ость", "ение", "ание", "ский", "оват", "ивать", "ывать", "тель", "изм", "ция", "к", "ушк", "еньк"],
    },
    "ar": {
        "pre": ["ال", "ب", "ك", "س", "ف", "م", "و"],
        "suf": ["ات", "ون", "ين", "ة", "تان", "تين", "كما", "هما", "ياً", "ية", "ان"],
    },
    "ja": {
        "pre": ["お", "ご"],
        "suf": ["さん", "ちゃん", "くん", "さま", "たち", "達", "的", "性", "風", "型", "ら"],
    },
    "ta": {
        "pre": [],
        "suf": ["கள்", "ங்கள்", "ர்", "ம்", "ஆல்", "இல்", "க்கு", "த்தல்", "மாக", "படு"],
    },
    "tk": {
        "pre": [],
        "suf": ["lar", "ler", "yň", "syz", "siz", "ly", "li", "lyk", "lik", "şyk", "çy", "daş"],
    },
    "zh": {
        "pre": ["第", "老", "超", "可", "微", "反"],
        "suf": ["们", "化", "性", "儿", "家", "者", "式", "度", "系"],
    },
    "pl": {
        "pre": ["nie", "prze", "bez", "roz", "pod", "współ", "nad", "między", "naj"],
        "suf": ["anie", "enie", "owy", "owa", "ek", "ka", "arz", "nik", "nia", "ność", "sko", "stwo", "acz"],
    },
    "ne": {
        "pre": ["अति", "अन", "उप", "वि", "पुन", "पर"],
        "suf": ["हरु", "पन", "वाद", "वादी", "पन्यता", "करण", "इयाँ", "हरूको"],
    },
}

# ---------------------------------------------------------------------------
# Cross-lingual equivalence classes
# ---------------------------------------------------------------------------

CROSS_EQUIV = {
    "PL": {
        "en": {"s", "es"},
        "tr": {"lar", "ler"},
        "de": {"en", "er", "e", "n"},
        "ru": {"ы", "и", "а", "я"},
        "ar": {"ات", "ون", "ين"},
        "ja": {"たち", "達"},
        "ta": {"கள்"},
        "tk": {"lar", "ler"},
        "zh": {"们"},
        "pl": {"y", "i", "e", "owie", "ami"},
        "ne": {"हरु"},
    },
    "NEG": {
        "en": {"un", "in", "im", "ir", "il", "non"},
        "de": {"un", "in"},
        "pl": {"nie"},
        "ru": {"не"},
        "tr": {"ma", "me"},
        "ta": {"இல்லா"},
        "tk": {"däl"},
        "zh": {"不", "沒", "没", "無", "无"},
        "ne": {"न"},
        "ja": {"不", "非"},
    },
    "FUT": {
        "ar": {"س", "سوف"},
        "tr": {"acak", "ecek"},
        "ru": {"буд"},
        "tk": {"jak", "jek"},
        "pl": {"będ"},
        "ne": {"ला"},
        "zh": {"将", "會"},
    },
    "PAST": {
        "de": {"ge"},
    },
    "SUPER": {
        "en": {"est"},
        "de": {"st"},
        "pl": {"naj"},
        "ru": {"ейш"},
        "zh": {"最"},
    },
    "COMP": {
        "en": {"er", "more"},
        "de": {"er", "mehr"},
        "pl": {"szy", "bardziej"},
        "ru": {"ее", "ше"},
        "tr": {"daha"},
        "zh": {"更"},
    },
    "DIM": {
        "en": {"let", "ling"},
        "tr": {"cık", "cik", "cuk", "cük"},
        "ru": {"ик", "ок", "очка", "еньк", "ушк"},
        "pl": {"ek", "ka", "ę", "ko"},
        "es": {"ito", "ita", "illo", "illa"},
        "zh": {"小"},
    },
    "PROG": {
        "en": {"ing"},
        "tr": {"yor", "ıyor", "iyor", "uyor", "üyor"},
        "ta": {"கொண்டிருக்க", "இருக்கிற"},
        "tk": {"ýar", "ýär"},
        "ja": {"ている", "てる"},
        "zh": {"着"},
        "ne": {"दै"},
    },
    "PAST_{SFX}": {
        "en": {"ed", "t"},
        "tr": {"dı", "di", "du", "dü", "tı", "ti", "tu", "tü"},
        "de": {"t", "en", "te"},
        "ru": {"л", "ла", "ло", "ли"},
        "ar": {"ت", "وا"},
        "ja": {"た"},
        "ta": {"ட்ட", "ந்த", "து"},
        "tk": {"dy", "di", "ty", "ti"},
        "zh": {"了", "過", "过"},
        "pl": {"ł", "ła", "li", "ły"},
        "ne": {"यो", "एँ", "यौ", "ए"},
    },
    "FUT_{SFX}": {
        "tr": {"acak", "ecek"},
        "tk": {"jak", "jek"},
        "zh": {"將"},
        "pl": {"będzie"},
        "ne": {"ला"},
    },
    "POSS": {
        "en": {"'s"},
        "tr": {"ım", "im", "um", "üm", "ımız", "imiz", "umuz", "ümüz", "ları"},
        "tk": {"ym", "im", "um", "üm", "lygyň"},
        "ru": {"его", "ее", "их"},
        "pl": {"jego", "jej"},
        "ne": {"को"},
    },
    "AGENT": {
        "en": {"er", "or", "ist"},
        "de": {"er", "erin"},
        "ru": {"тель", "щик", "ик"},
        "pl": {"arz", "nik", "owiec"},
        "tr": {"cı", "ci", "çu", "cü"},
        "tk": {"çy"},
        "zh": {"者", "家"},
    },
    "ADVERB": {
        "en": {"ly"},
        "de": {"erweise", "lich"},
        "pl": {"nie"},
        "ru": {"о"},
        "tr": {"ce", "ça"},
        "zh": {"地"},
    },
}
