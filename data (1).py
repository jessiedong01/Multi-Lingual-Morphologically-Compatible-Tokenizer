from itertools import islice

import requests
from datasets import load_dataset
from tokenizer_core.utils import *

WIKIMEDIA_CONFIGS = {
    "en": ("English", "20231101.en"),
    "de": ("German", "20231101.de"),
    "tr": ("Turkish", "20231101.tr"),
    "zh": ("Chinese", "20231101.zh"),
    "pl": ("Polish", "20231101.pl"),
    "tk": ("Turkmen", "20231101.tk"),
}


def load_wikiann_corpus(codes, per_lang=500):
    print("Loading corpus from Hugging Face datasets hub...")
    texts, langs = [], []
    for code, name in codes.items():
        print(f"-> Loading '{name}' ({code})...")
        try:
            dataset = load_dataset("wikiann", code, split='train', streaming=True)
            taken = 0
            for ex in dataset:
                if taken >= per_lang: break
                toks = ex.get('tokens', [])
                txt = " ".join(toks) if isinstance(toks, list) else str(toks)
                if not txt.strip(): continue
                if looks_like_redirect(txt): continue
                texts.append(txt)
                langs.append(code)
                taken += 1
        except Exception as e:
            print(f"Could not load data for {code}: {e}")
    if not texts:
        print("Corpus could not be loaded.")
    else:
        print("Corpus loading complete.")
    print("-" * 60)
    return texts, langs


def load_wikipedia_corpus(per_lang=500, configs=None):
    """
    Streams multilingual paragraphs for the target languages using the
    `wikimedia/wikipedia` dataset (Arrow files, streaming-friendly). If a
    particular language fails, falls back to live Wikipedia API sampling.
    """
    cfgs = configs or WIKIMEDIA_CONFIGS
    print("Loading multilingual corpus (streaming Wikipedia dumps)...")
    texts, langs = [], []
    used_api = False
    for code, (name, hf_config) in cfgs.items():
        print(f"-> Loading '{name}' ({hf_config})...")
        taken = 0
        try:
            dataset = load_dataset(
                "wikimedia/wikipedia",
                hf_config,
                split="train",
                streaming=True,
            )
            for ex in dataset:
                if taken >= per_lang:
                    break
                txt = ex.get("text", "")
                if not isinstance(txt, str):
                    continue
                txt = txt.strip()
                if not txt:
                    continue
                if looks_like_redirect(txt):
                    continue
                texts.append(txt)
                langs.append(code)
                taken += 1
        except Exception as e:
            print(f"Could not stream dumps for {code}: {e}")
            print(f"  Falling back to live Wikipedia API for {code}...")
            api_paras = fetch_wikipedia_api_paragraphs(code, per_lang)
            if api_paras:
                used_api = True
                texts.extend(api_paras)
                langs.extend([code] * len(api_paras))
                continue
            print(f"  Wikipedia API fallback failed for {code}.")
        if taken < per_lang:
            print(f"  Only collected {taken} paragraphs for {code}.")

    if not texts:
        print("Corpus could not be loaded.")
    else:
        if used_api:
            print("Corpus loaded (some languages via Wikipedia API fallback).")
        else:
            print("Corpus loading complete.")
    print("-" * 60)
    return texts, langs


def fetch_wikipedia_api_paragraphs(lang_code, per_lang=500, session=None):
    """Fetches clean paragraphs via the public Wikipedia API."""
    base_url = f"https://{lang_code}.wikipedia.org/w/api.php"
    sess = session or requests.Session()
    paragraphs = []
    seen_pages = set()
    while len(paragraphs) < per_lang:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "exlimit": "max",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 5,
        }
        try:
            resp = sess.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"    Wikipedia API error for {lang_code}: {exc}")
            break
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            continue
        for page_id, page in pages.items():
            if page_id in seen_pages:
                continue
            seen_pages.add(page_id)
            extract = page.get("extract", "")
            if not extract:
                continue
            for para in extract.split("\n"):
                para = para.strip()
                if not para:
                    continue
                if looks_like_redirect(para):
                    continue
                paragraphs.append(para)
                if len(paragraphs) >= per_lang:
                    break
            if len(paragraphs) >= per_lang:
                break
    if not paragraphs:
        print(f"    No paragraphs collected for {lang_code}.")
    return paragraphs
