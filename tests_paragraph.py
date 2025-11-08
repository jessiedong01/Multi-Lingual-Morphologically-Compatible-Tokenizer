#Minimal demo harness (run once per session)
from tokenizer import ScalableTokenizer
from linguistic_features import LinguisticModels
from constants import *
import utils

def make_tok(alpha=1.0, beta=0.5, **kwargs):
    # Sensible toy defaults
    tok = ScalableTokenizer(
        max_token_len=64,
        min_freq=1,          # keep everything for demos
        top_k_add=12,
        vocab_budget=None,   # no λ bisection for toys
        tau=0.03             # small length penalty
    )
    tok.set_feature_models(**kwargs)
    return tok

def print_tok(text, corpus_texts, corpus_langs, tok):
    print(text)
    print("-------------")
    for corpus_text, corpus_lang in zip(corpus_texts, corpus_langs):
        print(f"text: {corpus_text}")
        tokens = tok.tokenize(corpus_text, lang=corpus_lang)
        print(f"\t{tokens}")
    print()

#####
# PARAGRAPH LEVEL
#####

def lexicon_test():
    print("@@@@@ Lexicon test @@@@@")
    texts = ["I live in New York, but I wished I lived in San Francisco."]
    langs = ["en"]
    tok_without = make_tok()
    tok_with = make_tok(lexicon={"New York": 5.0, "San Francisco": 10.0})
    #enable_virtual_arcs(tok)
    tok_without.train(texts, langs, max_iterations=10, verbose=False)
    tok_with.train(texts, langs, max_iterations=10, verbose=False)
    print_tok("testing without lexicon", texts, langs, tok_without)
    print_tok("testing with lexicon ", texts, langs, tok_with)


    for i in range (11):
        tok_with_mod = make_tok(lexicon={"New York": 5.0, "San Francisco": 10.0 - i})
        #enable_virtual_arcs(tok)
        tok_with_mod.train(texts, langs, max_iterations=10, verbose=False)
        print("weight:", 10-i)
        print_tok("testing with lexicon ", texts, langs, tok_with_mod)

# What’s happening: “New York” is admitted as a virtual arc; additive lexicon reward makes it win over “New”+“York”.
# How to modify: Lower lexicon weight to see it split; remove virtual arcs to see it fail on tiny data.
# Why it works: Contextual reward + lattice access without needing stats.

def ne_gaz_test():
    print("@@@@@ Named Entity Gazzetter test @@@@@")

    texts = ["I'm in Berlin and I saw Angela Merkel on my tour."]
    langs = ["en"]
    tok_without = make_tok()
    tok_with = make_tok(ne_gaz={"LOC": {"Berlin": 10.0}, "PER": {"Angela Merkel": 10.0}})
    #enable_virtual_arcs(tok)
    tok_without.train(texts, langs, max_iterations=100, verbose=False)
    tok_with.train(texts, langs, max_iterations=100, verbose=False)

    print_tok("Without named entities", texts, langs, tok_without)
    print("With named entity:")
    print_tok("Berlin is an LOC and Angela Merkel is a PER, shouldn't be split", texts, langs, tok_with)

    txt_test = ["Ich bin ein Berliner", "qwejfksalskdBerlinasodq"]
    langs = ["de", "en"]
    tok_test = make_tok(ne_gaz={"LOC": {"Berlin": 10.0}, "PER": {"Angela Merkel": 10.0}})
    tok_test.train(txt_test, langs, max_iterations=100, verbose=False)
    print("Berlin with appendum")
    print("Berlin in the middle of a mesh")
    print_tok("Ich bin ein Berliner", txt_test, langs, tok_test)

    tok_with_more = make_tok(ne_gaz={"LOC": {"Berlin": 10.0, "Beijing": 10.0, "Tokyo": 10.0}, "PER": {"Angela Merkel": 10.0, "Jamie Dimon": 10.0}})
    texts_more = ["Jamie Dimon met with Angela Merkel in Berlin about the conflict between Beijing and Tokyo"]
    langs_more = ["en"]
    tok_with_more.train(texts_more, langs_more, max_iterations=100, verbose = False)
    print_tok("Beijing and Tokyo met with Angela Merkel in Jamie Dimon hometown conflict the", texts, langs, tok_with)
    print_tok("Beijing and Tokyo met with Angela Merkel in Jamie Dimon hometown conflict the", texts_more, langs_more, tok_with_more)

# What’s happening: “Berlin” is allowed and gets a small bonus.
# How to modify: Add more entity types; try a sentence with “Berlin” mid-string.
# Why it works: Same mechanism as lexicon.


def mwe_test():
    print("@@@@@ Multi Word Entity (MWE) test @@@@@")
    tok = make_tok(mwe={"San Jose"})
    #enable_virtual_arcs(tok)
    s = "He flew to San Jose yesterday."
    tok.train([s], ["en"], max_iterations=20, verbose=False)
    print(tok.tokenize(s, lang="en"))
    print_tok("testing MWE (multi-word entities), similar to lexicon", [s], ['en'], tok)

    tok_remove = make_tok()
    #enable_virtual_arcs(tok)
    txt = "He flew to San Jose yesterday."
    tok_remove.train([txt], ["en"], max_iterations=20, verbose=False)
    print(tok_remove.tokenize(txt, lang="en"))
    print_tok("testing MWE (multi-word entities), similar to lexicon", [txt], ['en'], tok_remove)

    tok_reinforced = make_tok(mwe={"San Jose": 10.0})
    #enable_virtual_arcs(tok)
    tok_reinforced.train([s], ["en"], max_iterations=20, verbose=False)
    print(tok_reinforced.tokenize(s, lang="en"))
    print_tok("testing MWE (multi-word entities), similar to lexicon", [s], ['en'], tok_reinforced)


# What’s happening: MWE list makes “San Jose” one token. Similar effect to the lexicon test.
# How to modify: Remove virtual arcs to see it split on tiny corpora; add a small lexicon score to reinforce.

def affix_test():
    print("@@@@@ Named Entity Gazzetter test @@@@@")

    texts = [#'re-' and 'un-' along with suffixes '-ed', '-tion', '-s', '-ly', and '-ness'
            "The renewed conversations sadly revealed his unhappiness",
             # prefixes 'dis-' and 'in-' are combined with suffixes '-ly', '-ing', and '-s'
             "Dishonestly, he was incapable of finding reasons for his actions.",
             # 'un-' and 'dis-' with the suffixes '-ed', '-ness', '-tion', '-ing', and '-s'
             "The unexpected quickness of the decision was disturbing to viewers."
             ]
    langs = ["en", "en", "en"]
    tok_without = make_tok(suffix_reward=0.0, prefix_reward=0.0, space_penalty=1.0)
    tok_with = make_tok(suffix_reward=5.0, prefix_reward=5.0, space_penalty=1.0)
    #enable_virtual_arcs(tok)
    tok_without.train(texts, langs, max_iterations=20, verbose=False)
    tok_with.train(texts, langs, max_iterations=20, verbose=False)

    print_tok("Without affix rewards", texts, langs, tok_without)
    print_tok("With affix rewards", texts, langs, tok_with)

    '''print("testing suffix parameters ---------------")
    for i in range (-10, 5, 2):
        tok_with_tests = make_tok(suffix_reward=5.0 - i, prefix_reward=5.0, space_penalty=1.0)
        #enable_virtual_arcs(tok)
        tok_with_tests.train(texts, langs, max_iterations=20, verbose=False)
        print("With suffix reward", 5-i)
        print_tok("With suffix reward", texts, langs, tok_with_tests)'''

    for i in range (-20, 5, 3):
        tok_with_tests = make_tok(suffix_reward=5.0, prefix_reward=5.0 - i, space_penalty=1.0)
        #enable_virtual_arcs(tok)
        tok_with_tests.train(texts, langs, max_iterations=20, verbose=False)
        print("With prefix reward", 5-i)
        print_tok("With prefix reward", texts, langs, tok_with_tests)

    
        
# What’s happening: We prioritize certain affixes. Notice that at high levels it makes really wonky mid-token splits.
# How to modify: Try some of the other sentences or modify the reward level.

def regex_test():
    print("@@@@@ Regex testing @@@@@")
    tok = make_tok()
    texts = ["Email me at alice@example.com and see www.catsbeingstupid.com on 2024-09-04.",
             "See cat at https://www.catsbeingstupid.com for $1,030.04"]
    langs = ["en", "en"]
    tok.train(texts, langs, max_iterations=50, verbose=True)
    print("vocabulary: ", tok.vocab)
    print_tok("we avoid splitting on certain regexes (emails)", texts, langs, tok)
# What’s happening: URL/email/date regex creates atomic protected spans; DP admits them even if not in vocab.
# So, what is happening here? Can you spot it? Try tokenize ["Email me at alice@example.com and see"]
# How to modify: Take a look at ScalableTokenizer.tokenize. Deep-dive from there.
# Why it works: Deterministic pattern match, no statistics needed.

def redirect_test():
    print("@@@@@ Redirect test @@@@@")
    tok = make_tok()
    txt = "#REDIRECT United States"
    tok.train([txt], ["en"], max_iterations=1, verbose=True)
    # Whether you filter or just flag depends on your downstream; show detection:
    value = bool(REDIRECT_TOKEN_RE.search(txt))
    print(f"testing redirect: Does {txt} have a redirect token?: {value}")
    print_tok("redirect_test", [txt], ['en'], tok)

    print("test spaced characters")

    tok_spaced = make_tok()
    txt_spaced = "#R E D I R E C T United States"
    tok_spaced.train([txt_spaced], ["en"], max_iterations=1, verbose=True)
    # Whether you filter or just flag depends on your downstream; show detection:
    value = bool(REDIRECT_TOKEN_RE.search(txt_spaced))
    print(f"testing redirect: Does {txt_spaced} have a redirect token?: {value}")
    print_tok("redirect_test", [txt_spaced], ['en'], tok_spaced)


    print("test multilingual")

    tok_multi = make_tok()
    txt_multi = "# umleiten United States"
    tok_multi.train([txt_multi], ["en"], max_iterations=1, verbose=True)
    # Whether you filter or just flag depends on your downstream; show detection:
    value = bool(REDIRECT_TOKEN_RE.search(txt_multi))
    print(f"testing redirect: Does {txt_multi} have a redirect token?: {value}")
    print_tok("redirect_test", [txt_multi], ['en'], tok_multi)

# What’s happening: Redirect regex fires on toy text. 
# Same as above, what is happening here?
# How to modify: Add spaced-letter variants (“R E D I R E C T”)—regex still catches.
# Why it works: Pure regex rule.

def morph_test():
    txt = ["I test the token candidate.", 
           "I tested the token candidate.", 
           "I am testing the token candidate.", 
           "I have tested the token candidate.", 
           "I will test the token candidate."]
    tok = make_tok()
    langs = ["en", "en", "en", "en", "en"]
    tok.train(txt, ["en", "en", "en", "en", "en"], max_iterations=20, verbose=True)
    print_tok("redirect_test", txt, langs, tok)

def affix_test_v2():
    print("@@@@@ Named Entity Gazzetter test @@@@@")

    texts = [#'re-' and 'un-' along with suffixes '-ed', '-tion', '-s', '-ly', and '-ness'
            "The renewed conversations sadly revealed his unhappiness",
             # prefixes 'dis-' and 'in-' are combined with suffixes '-ly', '-ing', and '-s'
             "Dishonestly, he was incapable of finding reasons for his actions.",
             # 'un-' and 'dis-' with the suffixes '-ed', '-ness', '-tion', '-ing', and '-s'
             "The unexpected quickness of the decision was disturbing to viewers."
             ]
    langs = ["en", "en", "en"]
    tok_without = make_tok(suffix_reward=0.0, prefix_reward=0.0, space_penalty=1.0)
    tok_with = make_tok(suffix_reward=1000, prefix_reward=1000, space_penalty=1.0)
    #enable_virtual_arcs(tok)
    tok_without.train(texts, langs, max_iterations=20, verbose=False)
    tok_with.train(texts, langs, max_iterations=20, verbose=False)

    print_tok("Without affix rewards", texts, langs, tok_without)
    print_tok("With affix rewards", texts, langs, tok_with)

if __name__ == '__main__':
    #lexicon_test()
    #ne_gaz_test()
    #mwe_test()
    #affix_test()
    #regex_test()
    #redirect_test()
    morph_test()
    #affix_test_v2()
