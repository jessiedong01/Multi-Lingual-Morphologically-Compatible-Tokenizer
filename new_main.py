import os
from tokenizer import ScalableTokenizer
from linguistic_features import LinguisticModels

# --- 1. Craft a Diverse, Multilingual Corpus ---
# This corpus is specifically designed to contain a variety of linguistic phenomena
# that our feature models can latch onto.
print("1. Using a feature-rich, multilingual corpus.")
corpus_texts = [
    # English Examples
    "Dr. Anya Sharma works at Google in New York.",         # <-- Has PER, ORG, LOC entities and an affix 'works'
    "The new tokenizers are running quickly.",              # <-- Has plural, progressive tense, and adverbial suffixes
    "San Francisco is a beautiful city.",                  # <-- Contains a Multi-Word Expression (MWE)
    "This is a test of the system's representations.",      # <-- Contains a nominalization suffix 'tion'

    # German Examples
    "Dr. Schmidt arbeitet bei der Technischen UniversitÃ¤t.", # <-- Has PER, ORG entities
    "Die Entwicklungen sind gut.",                         # <-- Has plural and nominalization suffixes ('en', 'ung')
    "Er ist nach Berlin gegangen.",                        # <-- Contains a known location and a past participle
    "Eine schnelle Weiterleitung wurde erstellt.",         # <-- Contains a keyword for filtering and a suffix 'ung'
    
    # Turkish Examples (to test cross-lingual features)
    "Arabalar yolda gidiyor.",                              # <-- Has plural 'lar' and progressive 'yor'
    "Turkiye'nin baskenti Ankara'dir."                      # <-- Contains known locations
]
corpus_langs = ["en", "en", "en", "en", "de", "de", "de", "de", "tr", "tr"]


# --- 2. Initialize the Tokenizer ---
# We give it a slightly larger budget to accommodate the richer corpus.
print("2. Initializing the ScalableTokenizer.")
tokenizer = ScalableTokenizer(
    max_token_len=13,
    min_freq=1,          # Set to 1 for this small demo corpus
    top_k_add=2,         # Add more promising candidates per iteration
    vocab_budget=100,     # Aim for a vocabulary of ~100 multi-character tokens
    tau=0.01
)


# --- 3. Build a Comprehensive Linguistic "Brain" ---
# This is the most critical step. We are providing a rich set of priors
# to guide the tokenizer's optimization process.
print("3. Priming the tokenizer with extensive linguistic models.")

# Lexicon: A dictionary of known "good" tokens and their scores (rewards).
lexicon = {
    "Google": 3.0, "Microsoft": 3.0, "â€™s": 0.5, "'s": 0.5,
    "tokenizers": 2.0, "representations": 2.0, "Entwicklungen": 2.0,
    "Weiterleitung": 2.0, "UniversitÃ¤t": 2.0
}

# Multi-Word Expressions: A set of phrases that should *always* be one token.
mwe_set = {"San Francisco", "New York"}

# Named Entity Gazetteer: A dictionary mapping entity types to known examples.
# This helps the tokenizer learn the *concept* of a person, location, etc.
ne_gazetteer = {
    "PER": {"Anya Sharma": 1.0, "Schmidt": 1.0},
    "LOC": {"Berlin": 1.0, "Ankara": 1.0, "Turkiye": 1.0},
    "ORG": {"Google": 1.0, "Technischen Universität": 1.0}
}

# Token Bigrams: Costs/rewards for transitions between token *classes*.
# Negative costs are rewards. This models the syntax of named entities.
token_bigrams = {
    # Reward sentences starting with a capital word.
    ("<BOS>", "InitCap"): -0.1,
    # Heavily reward sequences of capitalized words (e.g., "Anya" -> "Sharma").
    ("InitCap", "InitCap"): -0.1,
    # Reward a number following another number.
    ("NUM", "NUM"): -0.2,
    # Small penalty for a random lowercase word after a capitalized one.
    ("InitCap", "lower"): 0.00
}

# Apply all these models and tune the weights for linguistic features.
# A higher `mu_morph` means it will trust the morphology encoder more.
tokenizer.set_feature_models(
    lexicon=lexicon,
    mwe=mwe_set,
    ne_gaz=ne_gazetteer,
    token_bigram=token_bigrams,
    gamma_boundary=0.05,  # Penalty for changing token classes (e.g., InitCap -> lower)
    mu_morph=0.1,         # Weight for the morphology encoder score (crucial!)
    prefix_reward=0.05,   # Small reward for known prefixes
    suffix_reward=0.1     # Small reward for known suffixes (e.g., -ing, -ung)
)

# --- 4. Train the Tokenizer ---
# With the linguistic brain in place, the training will now be guided towards
# meaningful, human-like tokens.
print("\n4. Starting the training process...")
tokenizer.train(corpus_texts, corpus_langs, max_iterations=150)


# --- 5. Analyze the Results ---
print("\n" + "="*50)
print("--- 5. ANALYSIS OF LEARNED VOCABULARY ---")
print("="*50 + "\n")

# Extract the learned vocabulary, excluding single characters.
learned_vocab = sorted([tok for tok in tokenizer.vocab if len(tok) > 1])

# Check for specific, high-quality tokens we expected it to learn.
print("âœ… Named Entities and Multi-Word Expressions:")
expected_ne = ["Anya Sharma", "New York", "San Francisco", "Schmidt", "Google", "Berlin", "Ankara"]
found_ne = [tok for tok in expected_ne if tok in tokenizer.tok2id]
print(f"   -> Found: {found_ne}\n")

print("âœ… Morphologically Complex Words:")
expected_morph = ["tokenizers", "running", "quickly", "representations", "Entwicklungen", "Weiterleitung", "gegangen"]
found_morph = [tok for tok in expected_morph if tok in tokenizer.tok2id]
print(f"   -> Found: {found_morph}\n")

print("âœ… Common Function Words:")
expected_common = ["This", "is", "works", "at", "Die", "sind", "wurde"]
found_common = [tok for tok in expected_common if tok in tokenizer.tok2id]
print(f"   -> Found: {found_common}\n")


# --- 6. Test on New Sentences ---
print("\n" + "="*50)
print("--- 6. TOKENIZING NEW TEXT ---")
print("="*50 + "\n")

test_sentences = [
    ("Dr. Li Wei joined Google in San Francisco.", "en"),
    ("Die neuen Darstellungen sind beeindruckend.", "de"), # 'Darstellungen' is a new word with a known suffix
    ("O'zbekistonning poytaxti Toshkentdir.", "uz") # Uzbek, an unseen but related Turkic language
]

for sentence, lang in test_sentences:
    tokens = tokenizer.tokenize(sentence, lang=lang)
    print(f"   Input:  '{sentence}' ({lang})")
    print(f"   Output: {tokens}\n")
print(tokenizer.vocab)
