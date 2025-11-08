# CS197 Project: The secret life of Tokens
**Course Staff:** [Alexander Johansen](alrojo.github.io) (contact: arjo@stanford.edu)

Welcome, CS197 researchers! What if I told you that the giant, world-changing LLMs we use every day are built on a surprisingly rickety foundation? That before a model can write a poem or a line of code, it first feeds the text through a simple compressor (byte-pair encoding) that often chews up words into meaningless, inefficient byte-mush. This is the current state of tokenization, and it's one of the biggest bottlenecks holding back truly global and specialized AI.

This quarter, you're being let in on a secret. You will be among the first to work with a **brand-new, unpublished tokenizer**. Instead of just greedily merging bytes, this `ScalableTokenizer` tries to learn language the way a linguist might, using optimization and a rich understanding of morphology and grammar.

Does it work? We think so. Could it revolutionize how LLMs handle the world's 7,000 languages? Could it create smarter tokenizations for programming languages, or even a made-up language from a sci-fi novel? Maybe! But right now, it's just a promising piece of highly-experimental code.

That's where you come in. Your mission is to take this powerful-but-untested framework, kick the tires, find its breaking points, push it to its limits, and help forge it into something that could genuinely change the field. This is not a typical class project; it's a real research incubator. Let's get started.

---

## Course Structure & Timeline

This 10-week project is divided into two phases:
* **Weeks 1-2: The Bootcamp.** Everyone will complete the same intensive curriculum, working through all the foundational materials (notebooks 1, 4, 5 and PDFs 2, 3). This will give you a shared, deep understanding of the problem and our proposed solution.
* **Weeks 4-10: Specialized Project Track.** Armed with this knowledge, you will choose a specialized project to own for the rest of the quarter, culminating in a final presentation and report.

---

## Phase 1: The Bootcamp (Weeks 1-3)

#### Getting Started 🚀
Follow these steps to set up your project environment.
1. **Prerequisites**
    If you haven't already, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or Anaconda, which will help manage your project's software packages.

2. **Create and Activate the Environment**
    Open your terminal and run the following commands to create a dedicated environment for this project. This keeps its dependencies separate from your other projects.
    ```
    # Create an environment named 'cs197-tokenizer' with Python 3.10
    conda create --name cs197-tokenizer python=3.10

    # Activate the new environment
    conda activate cs197-tokenizer
    ```
3. **Install Dependencies**
    With your environment active, install the necessary libraries from the `requirements.txt` file.
    ```
    # Install all required packages
    pip install -r requirements.txt
    ```
4. **Begin the Project**
    You're all set! Start by launching Jupyter and opening the first notebook, `1_Introduction.ipynb`, to dive into the core concepts.
    ```
    jupyter notebook
    ```
**Quick Run**
Test by running the tokenizer in the `main.py` script. This file is configured with default values, which you can open and edit to experiment with different settings.
```python main.py```

**Week 1: Foundations & The State of the Art**
* **Objective:** Understand why tokenization is a critical and difficult problem, and learn how current SOTA models do it.
* **Materials:**
    * `1_Introduction.ipynb`: Core concepts of character encoding, tokenization, and BPE.
    * `2_Related_works.pdf`: A literature review of modern tokenization techniques. This document contains a collection of many new interested papers in the field of tokenization. Some are described and detailed in a related works fashion, others are merely listed for inspiration.
* **Assignment 1**: You are given a curated set of sentences in multiple languages (e.g., English, German, Japanese, Arabic). See `main.py` as example for generating data. Process these sentences using the tokenizers for three major new LLMs (Qwen3, Llama 3, DeepSeek, T5). Write a critical analysis of their outputs. Where do they produce clean, semantic tokens? Where do they fall back to "byte-mush"? Think about - what have they been trained on? What might they not have a lot of exposure to? Papers that go here: [Sennrich et al., 2015](https://arxiv.org/abs/1508.07909), [Kudo, 2018](https://arxiv.org/abs/1808.06226)

**Week 2: A New Paradigm - Theory & Practice**
Objective: Grasp the mathematical theory behind the ScalableTokenizer and immediately apply it in practice. (we will upload this part soon, hashing out some minor design bugs atm.)
* **Materials:**
    * `4_learn_scalable.ipynb`: A hands-on walkthrough of training the tokenizer and using its linguistic features.
    * `tests_paragraph.py`: Testing the ScalableTokenizer on a couple of paragraphs with certain parameters set.
    * `main.py`: A script to run on a small corpora (you can modify parameters inside)
* **Assignment 2:** Solve the problems outlined in `tests_paragraph.py`. Look through how tokens are added with the function `_find_best_new_tokens_batch`, in particular, the cost function used in the function. Now, try and add different parameters to push segmentations in certain ways. Expand to larger corpora, languages, utilize the `CROSS_EQUIV` (see `MorphologicalEncoder`).

**Week 3: Mastering the Machine - The Algorithm Deep Dive**
* **Objective:** Achieve a deep, implementer-level understanding of the core dynamic programming algorithm.
* **Materials:**
    * `3_Methods.pdf` combined with `5_Tokenizer_deep_dive.ipynb`: The detailed mathematical-foundations of the tokenizer. Warning: This is dense! The goal is not to memorize every equation, but to understand the core concepts of the cost function and optimization approach. I.e. if a word has a `-ing` ending, how would that impact the cost for the word `running` over `runnin` (a cutoff token).
    * `tokenizer.py`: A focused exploration of the `_dp_decode` method and its Viterbi-like logic.
* **Assignment 3:** For a short, simple sentence like "cats run.", manually trace the execution of the DP algorithm. Fill out the DP cost table and backpointers to show how the optimal path is found. This is a crucial exercise to prove you understand the engine's core.

---

## Phase 2: Specialized Project Tracks (Weeks 4-10)

Choose one of the following projects for your deep dive. Each project includes a motivation, a list of key skills you will develop, and a detailed week-by-week plan.

### Project 1: High-Performance Tokenization 🏎️
* **Motivation:** The current Python implementation is great for research but too slow for training on massive, web-scale corpora. To be practically useful, it needs to be orders of magnitude faster. This project is a classic software engineering challenge: find the bottlenecks and crush them.
* **Key Skills:** Software Engineering, Profiling & Optimization, Cython/Parallel Programming.
* **Weekly Breakdown:**
    * **Week 4:** Use a profiler (`cProfile`) to rigorously identify the slowest parts of the training pipeline. Hypothesize that the nested loops for substring counting in `_initialize_stats_and_vocab` are the primary bottleneck. Research optimization strategies.
    * **Week 5:** Begin re-implementing the substring counting logic. Your main tool will be **Cython**, which compiles Python-like code into highly efficient C. Focus on getting a basic, correct implementation working.
    * **Week 6:** Debug and validate your Cython implementation to ensure it produces identical counts to the original Python code. Start exploring **multiprocessing** to parallelize the analysis across multiple CPU cores.
    * **Week 7:** Highlight your initial speed-up results. Refine your parallelization strategy and integrate it cleanly with the main tokenizer class.
    * **Week 8:** Run benchmarks on progressively larger corpora (10MB, 100MB, 1GB). Measure the wall-clock time difference between the original and your optimized version.
    * **Week 9:** Analyze your results and create plots showing the performance scaling. Finalize your code and write the final report, detailing the techniques used and the achieved speed-up factor.
    * **Week 10: Final Presentation.**

### Project 2: Comparative Failure Analysis: ScalableTokenizer vs. BPE 🧐
* **Motivation:** Standard benchmarks (like compression rate) don't tell the whole story. We need to understand the qualitative differences between tokenizers. This project moves beyond simple metrics to identify the specific linguistic scenarios where BPE struggles and our approach excels (or vice-versa), providing crucial insights for future research.
* **Key Skills**: Linguistics, Critical Thinking, Experimental Design.
* **Weekly Breakdown**:
    * **Week 4:** Research and document the common failure modes of BPE. Focus on issues like its handling of morphology, numbers, and multilingual code-switching.
    * **Week 5:** Design and create a "challenge suite" of difficult texts. This should include sentences from morphologically rich languages (Turkish, German), text with noise (typos, social media slang), and sentences with complex numerical or logical structures.
    * **Week 6:** Train both the `ScalableTokenizer` and a standard BPE tokenizer (e.g., from Hugging Face `tokenizers`) on the same baseline corpus with the same vocabulary size.
    * **Week 7:** Present your challenge suite and initial comparative results. Get feedback on other potential linguistic corner cases to investigate.
    * **Week 8:** Run both trained tokenizers over your entire challenge suite. Meticulously collect and categorize the outputs, noting every instance where the tokenizations differ significantly.
    * **Week 9:** Write your final report. This will be a qualitative analysis, rich with examples, explaining why the tokenizers behaved differently and arguing which approach is linguistically superior in each case.
    * **Week 10: Final Presentation.**

### Project 3: Expand the Morphological Feature Bank 📚
* **Motivation:** The tokenizer's linguistic "knowledge" is currently limited to a handful of common affixes in a few languages. By systematically expanding this feature bank, you can dramatically improve its ability to identify meaningful sub-words, especially for languages with rich and complex morphology.
* **Key Skills:** Computational Linguistics, Data Curation, Python Scripting, excitement about languages.
* **Weekly Breakdown:**
    * **Week 4:** Research linguistic resources for morphological data. Excellent sources include Wiktionary, UniMorph, and academic papers on computational morphology. Select 5-10 new languages to focus on, prioritizing those with rich morphology (e.g., Finnish, Turkish, Swahili, Russian).
    * **Week 5:** Develop Python scripts to parse these resources and automatically extract high-quality lists of prefixes and suffixes for your target languages.
    * **Week 6:** Clean and curate the extracted affix data. Integrate this new, expanded data into the `AFFIXES` dictionary in `constants.py`.
    * **Week 7:** Present your expanded feature bank. Design an experiment to prove that your new features improve tokenization quality.
    * **Week 8:** Run your evaluation. Train two versions of the tokenizer—one with the original `AFFIXES` and one with your expanded set. Analyze the resulting vocabularies.
    * **Week 9:** Quantify the improvement. Do you see more morphemes (like "ung" in German or "lar" in Turkish) appearing as distinct tokens? Write a report detailing your data collection process and the impact of the new features.
    * **Week 10: Final Presentation.**

### Project 4: Integrate Syntactic Structure via Part-of-Speech (POS) Tagging 🧠
* **Motivation:** Our tokenizer understands morphology but is blind to syntax. It doesn't know a noun from a verb. By integrating Part-of-Speech (POS) information, we can teach it about basic grammar, preventing it from making ungrammatical merges (e.g., combining a verb with an unrelated noun). This is a step towards formalizing and incorporating richer linguistic structures.
* **Key Skills:** NLP Fundamentals (POS Tagging), API Integration, Algorithm Design.
* **Weekly Breakdown:**
    * **Week 4:** Research multilingual POS tagging libraries. The `Stanza` library from the Stanford NLP Group is an excellent choice. Set up a pipeline to pre-process the training corpus and annotate every word with its POS tag.
    * **Week 5:** Design modifications to the `LinguisticModels` class. The goal is to incorporate POS-based costs into the `additive_cost` function. A good starting point is to enhance the `token_bigram` model to consider POS transitions (e.g., rewarding an Adjective-Noun sequence).
    * **Week 6:** Implement your new POS-aware cost features. This will involve modifying the main DP loop in `_dp_decode` to access the pre-computed POS tags during segmentation.
    * **Week 7:** Present your design and initial implementation. Debug your feature integration and run small-scale tests.
    * **Week 8:** Train two tokenizers (with and without your POS features) on a large corpus. This will be computationally intensive.
    * **Week 9:** Analyze the results. Does the POS-aware tokenizer produce a vocabulary that better aligns with grammatical boundaries? Prepare a report with qualitative examples and quantitative analysis.
    * **Week 10:** Final Presentation.

### Project 5: Enhance Multilingual Cross-Training Consistency 🌐
* **Motivation**: The tokenizer should understand that a grammatical concept, like "plural," is the same across languages, even if it's spelled differently (English `-s`, German `-en`, Turkish `-lar`). This project aims to explicitly enforce this consistency, creating more robust cross-lingual representations.
* **Key Skills**: Linguistics, Representation Learning, NumPy/Linear Algebra.
* **Weekly Breakdown**:
    * **Week 4:** Systematically expand the `CROSS_EQUIV` dictionary in `constants.py`. Research and add equivalent morphemes for more grammatical categories (e.g., past tense, definiteness, cases) across at least 5 languages.
    * **Week 5:** Design a new regularization or loss term to be used within the `MorphologyEncoder.fit` method. The goal is to mathematically "pull" the vectors of equivalent morphemes (e.g., the vector for `suf:s` in English and the vector for `suf:en` in German) closer together.
    * **Week 6:** Implement this consistency loss using NumPy. This will involve modifying the matrix factorization step to incorporate your new objective.
    * **Week 7:** Present your expanded `CROSS_EQUIV` map and the mathematical formulation of your consistency term. Debug the implementation.
    * **Week 8:** Train two `MorphologyEncoder` models, one with and one without your new consistency term.
    * **Week 9:** Evaluate the results. Use metrics like cosine similarity to demonstrate that your new model learns more aligned representations for shared grammatical concepts. Write up your findings in the final report.
    * **Week 10: Final Presentation.**

### Project 6: Visualize the Dynamic Programming Lattice 📊
* **Motivation: The Viterbi algorithm in _dp_decode is powerful but abstract. By creating an interactive visualization, you can make this core algorithm transparent and build a deep, intuitive understanding of how the optimal tokenization is found. This is an invaluable tool for debugging, teaching, and analysis.
* **Key Skills: Algorithms (Dynamic Programming), Data Visualization, Front-End Development.
* **Weekly Breakdown:**
    * **Week 4:** Instrument the `_dp_decode` function to log its internal state, specifically the full DP cost table and the backpointer table, for a given sentence.
    * **Week 5:** Choose a visualization library. `Plotly` is excellent for Python-based interactive dashboards, while `D3.js` offers maximum flexibility for web-based visualizations. Create a static plot of the lattice for a very short sentence.
    * **Week 6:** Build the core interactive components. A user should be able to input a sentence, and your tool should generate the full lattice graph, with nodes representing character boundaries and edges representing potential tokens.
    * **Week 7:** Demonstrate your prototype. Add features to display the cost on each edge (token) and highlight the single lowest-cost path found by the algorithm.
    * **Week 8:** Refine the UI/UX. Add tooltips, color-coding, and clear explanations to make the visualization accessible to someone unfamiliar with the algorithm.
    * **Week 9:** Finalize your visualization tool. Record a video walkthrough and write a report explaining how your tool can be used to understand the tokenizer's behavior on complex sentences.
    * **Week 10: Final Presentation.**
