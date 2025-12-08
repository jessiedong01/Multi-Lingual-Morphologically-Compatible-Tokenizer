"""
UniSeg Data Loader

Loads morpheme boundaries, prefixes, and suffixes from UniSegments JSONL files.
This module provides the data foundation for all morpheme-based rewards in the tokenizer.

Usage:
    from tokenizer_core.uniseg_loader import UniSegLoader
    
    loader = UniSegLoader("data/uniseg_word_segments")
    loader.load_language("en")
    
    # Get boundaries for a word
    bounds = loader.get_boundaries("walking", "en")  # {4}
    
    # Get affixes
    prefixes = loader.get_prefixes("en")  # {'un', 're', 'pre', ...}
    suffixes = loader.get_suffixes("en")  # {'ing', 'ed', 'er', ...}
"""

import json
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List
from dataclasses import dataclass, field


# Language code mapping (ISO 639-1 to ISO 639-3)
LANG_CODE_MAP = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "pl": "pol",
    "cs": "ces",
    "hu": "hun",
    "fi": "fin",
    "sv": "swe",
    "hi": "hin",
    "bn": "ben",
    "mr": "mar",
    "ml": "mal",
    "kn": "kan",
    "ta": "tam",
    "tr": "tur",
    "nl": "nld",
    "da": "dan",
    "no": "nor",
    "ro": "ron",
    "uk": "ukr",
    "hr": "hrv",
    "sr": "srp",
    "bg": "bul",
    "sk": "slk",
    "sl": "slv",
    "et": "est",
    "lv": "lav",
    "lt": "lit",
    "ca": "cat",
    "gl": "glg",
    "eu": "eus",
    "cy": "cym",
    "ga": "gle",
    "mt": "mlt",
    "sq": "sqi",
    "mk": "mkd",
    "bs": "bos",
    "is": "isl",
    "fa": "fas",
    "ar": "ara",
    "he": "heb",
    "th": "tha",
    "vi": "vie",
    "id": "ind",
    "ms": "msa",
    "tl": "tgl",
    "sw": "swa",
    "af": "afr",
    "am": "amh",
    "ja": "jpn",
    "ko": "kor",
    "zh": "zho",
    "mn": "mon",
    "ka": "kat",
    "hy": "hye",
    "az": "aze",
    "kk": "kaz",
    "uz": "uzb",
    "ky": "kir",
    "tg": "tgk",
    "tk": "tuk",
}


@dataclass
class UniSegData:
    """Container for UniSeg data for a single language."""
    boundaries: Dict[str, Set[int]] = field(default_factory=dict)  # word -> boundary positions
    prefixes: Set[str] = field(default_factory=set)
    suffixes: Set[str] = field(default_factory=set)
    stems: Set[str] = field(default_factory=set)
    word_count: int = 0
    
    def has_data(self) -> bool:
        return self.word_count > 0


class UniSegLoader:
    """Loader for UniSegments JSONL data files."""
    
    def __init__(self, root_path: str | Path):
        """Initialize the loader.
        
        Args:
            root_path: Path to the uniseg_word_segments directory
        """
        self.root = Path(root_path) if root_path else None
        self._data: Dict[str, UniSegData] = {}
        self._loaded_langs: Set[str] = set()
    
    def _get_lang_dir(self, lang: str) -> Optional[Path]:
        """Get the directory path for a language."""
        if self.root is None:
            return None
        
        # Try ISO 639-3 code first
        lang_3 = LANG_CODE_MAP.get(lang, lang)
        
        candidates = [
            self.root / lang_3,
            self.root / lang,
            self.root / lang.lower(),
        ]
        
        for path in candidates:
            if path.exists() and path.is_dir():
                return path
        
        return None
    
    def load_language(self, lang: str) -> bool:
        """Load UniSeg data for a language from JSONL files.
        
        Args:
            lang: ISO 639-1 language code (e.g., 'en', 'de')
            
        Returns:
            True if data was loaded successfully
        """
        if lang in self._loaded_langs:
            return lang in self._data
        
        self._loaded_langs.add(lang)
        
        lang_dir = self._get_lang_dir(lang)
        if lang_dir is None:
            return False
        
        data = UniSegData()
        
        # ONLY load MorphoLex.jsonl (not MorphyNet or other files)
        morpholex_file = lang_dir / "MorphoLex.jsonl"
        if morpholex_file.exists():
            self._load_jsonl_file(morpholex_file, data)
        else:
            # Fallback: try first available jsonl if no MorphoLex
            for jsonl_file in lang_dir.glob("*.jsonl"):
                self._load_jsonl_file(jsonl_file, data)
                break  # Only load one file
        
        if data.has_data():
            self._data[lang] = data
            return True
        
        return False
    
    def _load_jsonl_file(self, path: Path, data: UniSegData) -> None:
        """Load a single JSONL file into the data container."""
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        self._process_record(record, data)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception:
            pass
    
    def _process_record(self, record: dict, data: UniSegData) -> None:
        """Process a single JSONL record."""
        word = record.get("word", "")
        if not word:
            return
        
        word_lower = word.lower()
        data.word_count += 1
        
        # Extract boundaries
        boundaries = record.get("boundaries", [])
        if boundaries:
            data.boundaries[word_lower] = set(int(b) for b in boundaries)
        
        # Extract morphemes by type from segments
        segments = record.get("segments", [])
        for seg in segments:
            seg_type = seg.get("type", "").lower()
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            
            if start >= end or end > len(word):
                continue
            
            morpheme = word_lower[start:end]
            
            # Only keep morphemes of reasonable length
            if len(morpheme) < 2:
                continue
            
            if seg_type == "prefix":
                data.prefixes.add(morpheme)
            elif seg_type == "suffix":
                data.suffixes.add(morpheme)
            elif seg_type in ("stem", "root"):
                data.stems.add(morpheme)
    
    def get_boundaries(self, word: str, lang: str) -> Optional[Set[int]]:
        """Get morpheme boundaries for a word.
        
        Args:
            word: The word to look up
            lang: Language code
            
        Returns:
            Set of boundary positions, or None if not found
        """
        if lang not in self._loaded_langs:
            self.load_language(lang)
        
        data = self._data.get(lang)
        if data is None:
            return None
        
        return data.boundaries.get(word.lower())
    
    def get_prefixes(self, lang: str) -> Set[str]:
        """Get all prefixes extracted from UniSeg for a language."""
        if lang not in self._loaded_langs:
            self.load_language(lang)
        
        data = self._data.get(lang)
        return data.prefixes if data else set()
    
    def get_suffixes(self, lang: str) -> Set[str]:
        """Get all suffixes extracted from UniSeg for a language."""
        if lang not in self._loaded_langs:
            self.load_language(lang)
        
        data = self._data.get(lang)
        return data.suffixes if data else set()
    
    def get_stems(self, lang: str) -> Set[str]:
        """Get all stems/roots extracted from UniSeg for a language."""
        if lang not in self._loaded_langs:
            self.load_language(lang)
        
        data = self._data.get(lang)
        return data.stems if data else set()
    
    def get_valid_positions(self, word: str, lang: str) -> Set[int]:
        """Get all valid boundary positions for a word.
        
        This includes:
        - Position 0 (word start)
        - Position len(word) (word end)
        - Internal morpheme boundaries from UniSeg
        
        Args:
            word: The word to check
            lang: Language code
            
        Returns:
            Set of valid boundary positions
        """
        positions = {0, len(word)}
        
        boundaries = self.get_boundaries(word, lang)
        if boundaries:
            positions.update(boundaries)
        
        return positions
    
    def is_morpheme_aligned(self, word: str, start: int, end: int, lang: str) -> bool:
        """Check if a span (start, end) is morpheme-aligned.
        
        A span is aligned if BOTH start and end are at valid boundary positions.
        
        Args:
            word: The full word
            start: Start position of span
            end: End position of span
            lang: Language code
            
        Returns:
            True if both boundaries are valid
        """
        valid = self.get_valid_positions(word, lang)
        return start in valid and end in valid
    
    def get_stats(self, lang: str) -> Dict[str, int]:
        """Get statistics about loaded data for a language."""
        if lang not in self._loaded_langs:
            self.load_language(lang)
        
        data = self._data.get(lang)
        if data is None:
            return {"words": 0, "prefixes": 0, "suffixes": 0, "stems": 0}
        
        return {
            "words": len(data.boundaries),
            "prefixes": len(data.prefixes),
            "suffixes": len(data.suffixes),
            "stems": len(data.stems),
            "total_entries": data.word_count,
        }
    
    def word_to_morphemes(self, word: str, lang: str) -> Optional[List[str]]:
        """Split a word into its morphemes based on UniSeg boundaries.
        
        Args:
            word: The word to split
            lang: Language code
            
        Returns:
            List of morpheme strings, or None if word not in database
        """
        boundaries = self.get_boundaries(word, lang)
        if boundaries is None:
            return None
        
        word_lower = word.lower()
        positions = sorted([0] + list(boundaries) + [len(word_lower)])
        
        morphemes = []
        for i in range(len(positions) - 1):
            morpheme = word_lower[positions[i]:positions[i+1]]
            if morpheme:
                morphemes.append(morpheme)
        
        return morphemes


# Convenience function
def load_uniseg(root_path: str | Path, lang: str) -> UniSegLoader:
    """Load UniSeg data for a language.
    
    Args:
        root_path: Path to uniseg_word_segments directory
        lang: Language code
        
    Returns:
        Loaded UniSegLoader instance
    """
    loader = UniSegLoader(root_path)
    loader.load_language(lang)
    return loader

