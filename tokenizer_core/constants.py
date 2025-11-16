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

import string

PUNCT_SET = set(string.punctuation) | {
    # ==== Already Given ======================================================================
    "«", "»", "„", "“", "”", "—", "–", "…", "·",
    "：", "；", "！", "？", "。", "、",
    "「", "」", "『", "』",

    # ==== Common Latin / European extensions =================================================
    "‹", "›",                   # Single-angle quotation
    "‚", "‟",                   # Additional quotes
    "–", "—", "―",              # Dashes (n-dash, m-dash, horiz. bar)
    "•", "‣",                   # bullets
    "§",                        # section sign
    "¶",                        # paragraph
    "°",                        # degree sign
    "′", "″",                   # prime/double prime

    # ==== Spanish ============================================================================
    "¿", "¡",                   # inverted Q/E
 
    # ==== Polish / Central-European ==========================================================
    "’",                        # apostrophe variant (Polish/etc.)
    "”", "„",                   # alt quotes

    # ==== Slavic / Cyrillic ==================================================================
    "—", "–",                   # Russian dashes (dup but allowed)

    # ==== CJK (Chinese / Japanese) ===========================================================
    "＞", "＜",                 # fullwidth angle
    "，", "．",                 # fullwidth comma/period
    "！", "？",                 # fullwidth ! ?
    "；", "：",                 # fullwidth ; :
    "（", "）", "［", "］", "【", "】", "｛", "｝",  # brackets
    "﹁", "﹂", "﹃", "﹄",      # corner quotes
    "ー",                       # long mark (JP)

    # ==== Korean-relevant CJK symbols (often appear in mixed corpora) ========================
    "※",                       # reference mark

    # ==== Arabic / Persian ===================================================================
    "،",                        # comma
    "؛",                        # semicolon
    "؟",                        # question mark
    "ـ",                        # kashida (elongation)
    "«", "»",                   # angle quotes (Arabic typography)

    # ==== Devanagari (Nepali / Hindi) ========================================================
    "।",                        # danda (full stop)
    "॥",                        # double danda

    # ==== Tamil ===========================================================================
    "।",                        # Tamil sometimes uses Devanagari danda
    "௹",                        # Rupee sign
    "௳", "௴", "௵", "௶", "௷",   # Tamil numeric markers (punct-like)

    # ==== Misc Unicode Punctuation ==========================================================
    "†", "‡",                   # daggers
    "※",                        # reference mark
    "⸺", "⸻",                   # long dash variants
    "〃",                       # ditto mark
    "〜",                       # Japanese wave dash
    "〝", "〞",                 # double corner quotes
    "〟",                       # low corner quote
    "﹏",                       # underline punctuation
}


# ---------------------------------------------------------------------------
# Affix inventories
# ---------------------------------------------------------------------------

AFFIXES = {
    "en": {
        "pre": [
            "re", "un", "in", "im", "dis", "mis", "non",
            "over", "under", "pre", "post", "anti",
        ],  # NEG: un, in, im, non
        "suf": [
            "ing",   # PROG
            "ed", "t",   # PAST_{SFX}
            "er", "ers",   # AGENT/COMP
            "ly",         # ADVERB
            "ness", "ment", "tion", "sion", "ous",
            "able", "less", "ful", "hood", "ity", "ive",
            "s", "es",     # PL
            "est",         # SUPER
            "or", "ist",   # AGENT
            "let", "ling", # DIM
        ],
    },

    "de": {
        "pre": [
            "be", "ge", "ent", "ver", "zer", "er", "ur", "miss",
            "un", "in",   # NEG
        ],
        "suf": [
            "ung", "en", "er", "heit", "keit", "isch", "lich", "erei",
            "lein", "chen", "schaft",
            "st",           # SUPER
            "er", "mehr",   # COMP
            "t", "te",      # PAST_{SFX}
        ],
    },

    "tr": {
        "pre": [
            # Turkish is overwhelmingly suffixing; NEG prefix not productive
        ],
        "suf": [
            "ler", "lar",                    # PL
            "lık", "lik", "luk", "lük",      # relational noun
            "cı", "ci", "cu", "cü",          # AGENT
            "sız", "siz", "suz", "süz",      # NEG (privative)
            "dan", "den", "tan", "ten",      # abl
            "yor", "ıyor", "iyor", "uyor", "üyor",   # PROG
            "dı", "di", "du", "dü", "tı", "ti", "tu", "tü",   # PAST_{SFX}
            "muş", "miş", "müş", "muş",      # reported past
            "acak", "ecek",                  # FUT / FUT_{SFX}
            "daha",                          # COMP (quasi-bound)
        ],
    },

    "ru": {
        "pre": [
            "по", "про", "не", "без", "раз", "под", "пред", "над",
            "сверх", "вы", "за", "со", "при",
        ],  # NEG: не
        "suf": [
            "ник", "чик", "щик", "ость", "ение", "ание", "ский",
            "оват", "ивать", "ывать", "тель", "изм", "ция",
            "к", "ушк", "еньк",  # DIM
            "ейш",   # SUPER
            "ее", "ше",  # COMP
            "л", "ла", "ло", "ли",  # PAST_{SFX}
            "тель", "щик", "ик",   # AGENT
            "о",                   # ADVERB
        ],
    },

    "ar": {
        "pre": [
            "ال", "ب", "ك", "س", "ف", "م", "و",
            "لا", "غير", "عدم", "ما",  # NEG (non-affixal particles included)
        ],
        "suf": [
            "ات", "ون", "ين", "ة", "تان", "تين", "كما", "هما",
            "ياً", "ية", "ان",
            "ت", "وا",       # PAST_{SFX}
            # FUT is prefixed س/سوف (already in pre)
        ],
    },

    "ja": {
        "pre": ["お", "ご"],
        "suf": [
            "さん", "ちゃん", "くん", "さま",
            "たち", "達",             # PL
            "的", "性", "風", "型", "ら",
            "ている", "てる",         # PROG
            "最", "一番",             # SUPER (bound-like)
            "者", "家",               # AGENT
            "不", "非",               # NEG (kanji prefixes)
            "に", "く",               # ADVERB
            "た",                     # PAST_{SFX}
        ],
    },

    "ta": {
        "pre": [],
        "suf": [
            "கள்", "ங்கள்",       # PL
            "ர்", "ம்",
            "ஆல்", "இல்", "க்கு",
            "த்தல்",
            "மாக",                # ADV
            "படு",
            "கொண்டிருக்க", "இருக்கிற",  # PROG
            "இல்லா",              # NEG
            "ட்ட", "ந்த", "து",   # PAST_{SFX}
        ],
    },

    "tk": {
        "pre": [],
        "suf": [
            "lar", "ler",        # PL
            "yň", "niň",         # POSS
            "syz", "siz",        # NEG/privative
            "ly", "li",          # relational
            "lyk", "lik",        # abstract
            "şyk",
            "çy",                # AGENT
            "daş",
            "jak", "jek",        # FUT_{SFX}
            "ýar", "ýär",        # PROG
            "dy", "di", "ty", "ti",   # PAST_{SFX}
        ],
    },

    "zh": {
        "pre": [
            "第", "老", "超", "可", "微", "反",
            "将", "會", "未",   # FUT/NEG markers
        ],
        "suf": [
            "们",   # PL
            "化", "性",
            "儿",   # DIM
            "家", "者",  # AGENT
            "式", "度", "系",
            "地",        # ADVERB
            "了", "過", "过",   # PAST/aspect
            "最", "更",          # SUPER/COMP
        ],
    },

    "pl": {
        "pre": [
            "nie",      # NEG
            "prze", "bez", "roz", "pod",
            "współ", "nad", "między",
            "naj",   # SUPER
            "będ",  # future stem
        ],
        "suf": [
            "anie", "enie",
            "owy", "owa",
            "ek", "ka", "ę", "ko",   # DIM
            "arz", "nik", "owiec",   # AGENT
            "nia", "ność",
            "sko", "stwo",
            "acz",
            "y", "i", "e", "owie", "ami",   # PL
            "szy", "bardziej",              # COMP
            "ł", "ła", "li", "ły",          # PAST_{SFX}
            "naj",                          # SUPER
        ],
    },

    "ne": {
        "pre": [
            "अति", "अन", "उप", "वि", "पुन", "पर",
            "न",   # NEG
        ],
        "suf": [
            "हरु",                 # PL
            "पन", "वाद", "वादी", "पन्यता",
            "करण",
            "इयाँ", "हरूको",
            "दै",                  # PROG
            "यो", "एँ", "यौ", "ए",    # PAST_{SFX}
            "ला",                 # FUT / FUT_{SFX}
            "को",                 # POSS
        ],
    },

    "bn": {
        "pre": [
            "অ", "অন", "অতি", "অধি", "উপ", "দুর", "অপ", "নির", "সু", "অধো",
            "প্রতি", "পুনঃ",
        ],
        "suf": [
            "রা", "গুলি", "গুলো",            # PL / collectivizers
            "তা", "ত্ব",                    # abstract nouns
            "শালী", "বাদ", "বৎ",             # relational/adjectival
            "কারী", "ওয়ালা",                # AGENT-like
            "চ্ছি", "চ্ছ", "ছে",             # PROG markers (finite)
            "লাম", "লেন", "লে", "লো",        # PAST tense endings
            "বে",                           # FUT ending
            "র", "এর", "দের",               # possessive endings
            "ভাবে",                        # adverbializer
        ],
    },


    "es": {
        "pre": [
            "in", "im", "ir", "i",     # NEG-ish
            "des",                     # NEG
            "re", "pre", "pos", "anti",
        ],
        "suf": [
            "s", "es",                 # PL
            "ito", "ita", "illo", "illa",  # DIM
            "ísimo",                     # SUPER
            "mente",                     # ADVERB
            "dor", "dora", "ero", "era", "ista",   # AGENT
            "ando", "iendo",             # PROG
            "é", "ó", "aste", "aron", "í", "ió",
            "iste", "ieron",             # PAST_{SFX}
            "ré", "rás", "rá", "remos", "réis", "rán",   # FUT_{SFX}
            "más",                        # COMP (analytic)
        ],
    },
}

# ---------------------------------------------------------------------------
# Cross-lingual equivalence classes
# ---------------------------------------------------------------------------

CROSS_EQUIV = {
    "PL": {
        "en": {"s", "es"},
        "tr": {"lar", "ler"},
        "de": {"e", "er", "en", "n", "s"},
        "ru": {"ы", "и", "а", "я"},
        "ar": {"ات", "ون", "ين"},
        "ja": {"たち", "達"},  # human/animate collectivizer
        "ta": {"கள்"},
        "tk": {"lar", "ler"},
        "zh": {"们"},
        "pl": {"y", "i", "e", "owie", "ami"},
        "ne": {"हरु"},
        "es": {"s", "es"},
        "bn": {"রা", "গুলি", "গুলো"},

    },

    "NEG": {
        "en": {"un", "in", "im", "ir", "il", "non"},
        "de": {"un", "in"},
        "pl": {"nie"},
        "ru": {"не"},
        "tr": {"ma", "me"},
        "ta": {"இல்லா"},
        "tk": {"däl"},
        "zh": {"不", "沒", "没", "無", "无", "未"},
        "ne": {"न"},
        "ja": {"不", "非", "ない", "じゃない", "ぬ"},
        "es": {"in", "im", "ir", "i", "des", "a"},  # des- very productive; a- before certain stems
        "ar": {"لا", "غير", "عدم", "ما"},
        "bn": {"অ", "অন", "নি", "নির"},

    },

    "FUT": {
        "ar": {"س", "سوف"},
        "tr": {"acak", "ecek"},   # also in FUT_{SFX}
        "ru": {"буд"},           # future auxiliary stem
        "tk": {"jak", "jek"},    # also in FUT_{SFX}
        "pl": {"będ"},           # periphrastic future stem
        "ne": {"ला"},            # enclitic-like future marker
        "zh": {"将", "會", "将"},  # variant forms included
        # (en/de/ja/es typically lack bound future morphology; omitted)
        "bn": {"বে"},

    },

    "PAST": {
        "de": {"ge"},   # participle prefix
        # (others handled in PAST_{SFX})
    },

    "SUPER": {
        "en": {"est"},
        "de": {"st"},
        "pl": {"naj"},
        "ru": {"ейш"},
        "zh": {"最"},
        "tr": {"en"},
        "ja": {"最", "一番"},
        "es": {"ísimo"},  # absolute superlative
        # (tk uses analytic 'iň' = superlative particle; include below for comparability)
        "tk": {"iň"},
        "ta": {"மிகவும்", "அதிகம்"},  # common superlative adverbs (analytic but bound-like usage)
    },

    "COMP": {
        "en": {"er", "more"},
        "de": {"er", "mehr"},
        "pl": {"szy", "bardziej"},
        "ru": {"ее", "ше"},
        "tr": {"daha"},
        "zh": {"更"},
        "tk": {"rak", "rek"},  # Turkic comparative suffix
        "ja": {"もっと"},
        "es": {"más"},
        "ta": {"மேலும்"},      # common comparative adverb
    },

    "DIM": {
        "en": {"let", "ling"},  # (y) is too lexical/irregular; omitted
        "tr": {"cık", "cik", "cuk", "cük"},
        "ru": {"ик", "ок", "очка", "еньк", "ушк"},
        "pl": {"ek", "ka", "ę", "ko"},
        "es": {"ito", "ita", "illo", "illa"},
        "zh": {"小"},
        "de": {"chen", "lein"},
        "ja": {"ちゃん", "こ"},  # -こ as 子 suffix (productive in compounds)
    },

    "PROG": {
        "en": {"ing"},
        "tr": {"yor", "ıyor", "iyor", "uyor", "üyor"},
        "ta": {"கொண்டிருக்க", "இருக்கிற"},
        "tk": {"ýar", "ýär"},
        "ja": {"ている", "てる"},
        "zh": {"着"},
        "ne": {"दै"},
        "es": {"ando", "iendo"},  # -ndo gerunds with estar
        # (de/ru/pl/ar typically analytic/aspectual; no bound progressive suffix)
        "bn": {"ছে", "চ্ছি", "চ্ছ"},

    },

    "PAST_{SFX}": {
        "en": {"ed", "t"},
        "tr": {"dı", "di", "du", "dü", "tı", "ti", "tu", "tü"},
        "de": {"t", "en", "te"},
        "ru": {"л", "ла", "ло", "ли"},
        "ar": {"ت", "وا"},      # person/number past endings
        "ja": {"た"},
        "ta": {"ட்ட", "ந்த", "து"},
        "tk": {"dy", "di", "ty", "ti"},
        "zh": {"了", "過", "过"},  # aspect/past-like markers
        "pl": {"ł", "ła", "li", "ły"},
        "ne": {"यो", "एँ", "यौ", "ए"},
        "es": {"é", "ó", "aste", "aron", "í", "ió", "iste", "ieron"},  # preterite set (representative)
        "bn": {"লাম", "লেন", "লে", "লো"},

    },

    "FUT_{SFX}": {
        "tr": {"acak", "ecek"},
        "tk": {"jak", "jek"},
        "zh": {"將", "将"},
        "pl": {"będzie"},
        "ne": {"ला"},
        "es": {"ré", "rás", "rá", "remos", "réis", "rán"},
        "bn": {"বে"},

    },

    "POSS": {
        "en": {"'s"},
        "tr": {"ım", "im", "um", "üm", "ımız", "imiz", "umuz", "ümüz", "ları", "leri"},
        "tk": {"ym", "im", "um", "üm", "nyň", "niň"},
        "ru": {"его", "ее", "их", "ин", "ов", "ев", "ын"},  # possessive adjective suffixes
        "pl": {"jego", "jej"},
        "ne": {"को"},
        "zh": {"的"},
        "de": {"s"},   # genitive -s for many mascul./neuter nouns
        "ja": {"の"},   # possessive/attributive particle
        "es": {"de"},  # analytic linker; included for cross-ling equivalence
        "ar": {"ه", "ها", "هم", "هن", "كما", "كم", "كن", "نا", "ي"},
        "bn": {"র", "এর", "দের"},

    },

    "AGENT": {
        "en": {"er", "or", "ist"},
        "de": {"er", "erin"},
        "ru": {"тель", "щик", "ик"},
        "pl": {"arz", "nik", "owiec"},
        "tr": {"cı", "ci", "çu", "cü"},
        "tk": {"çy"},
        "zh": {"者", "家"},
        "ja": {"者", "家"},
        "es": {"dor", "dora", "ero", "era", "ista"},
        "ar": {"م"},  # مُـ- (mu-) active-participle prefix; written here as base consonant
        "bn": {"কারী", "ওয়ালা"},

    },

    "ADVERB": {
        "en": {"ly"},
        "de": {"erweise", "lich"},
        "pl": {"nie"},
        "ru": {"о"},
        "tr": {"ce", "ça"},
        "zh": {"地"},
        "ja": {"に", "く"},    # na-adj → に; i-adj → く
        "es": {"mente"},
        "ar": {"اً"},         # tanwīn fatḥ for adverbial/acc. use
        "ta": {"ஆக"},        # -āka adverbializer
        "bn": {"ভাবে"},

    },
}

# ---------------------------------------------------------------------------
# Overrides from UniSeg-derived resources
# ---------------------------------------------------------------------------
from pathlib import Path
import json

_PACKAGE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _PACKAGE_DIR / "data"
if not _DATA_DIR.exists():
    _DATA_DIR = _PACKAGE_DIR.parent / "data"
_UNISEG_AFFIXES = _DATA_DIR / "uniseg_affixes.json"
_UNISEG_CROSS = _DATA_DIR / "uniseg_cross_equiv.json"

if _UNISEG_AFFIXES.exists():
    data = json.loads(_UNISEG_AFFIXES.read_text(encoding="utf-8"))
    for lang, buckets in data.items():
        pre = buckets.get("pre", [])
        suf = buckets.get("suf", [])
        AFFIXES[lang] = {
            "pre": pre,
            "suf": suf,
        }

if _UNISEG_CROSS.exists():
    data = json.loads(_UNISEG_CROSS.read_text(encoding="utf-8"))
    for morph_type, lang_map in data.items():
        if morph_type not in CROSS_EQUIV:
            CROSS_EQUIV[morph_type] = {}
        target = CROSS_EQUIV[morph_type]
        for lang, items in lang_map.items():
            target[lang] = set(items)