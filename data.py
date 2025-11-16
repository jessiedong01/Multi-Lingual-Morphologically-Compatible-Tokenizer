from datasets import load_dataset
from tokenizer_core.utils import *

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
