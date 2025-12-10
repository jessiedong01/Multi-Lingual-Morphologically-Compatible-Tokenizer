# Presentation Plan — Morphology-Aligned Multilingual Tokenizer

Audience: technical (ML/NLP). Duration target: ~12–13 minutes. Emphasis: cross-lingual morphology and morphology-aligned tokenization (skip POS). Each slide lists format, intent, visuals, and a speaker script.

## 1) Title / Hook
- **Format:** Clean title + subtitle; one hero visual split side-by-side (baseline vs morph-aligned) with a short caption; minimal text (≤2 bullets under the visual).
- **Content layout:** 
  - Title: “Morphology-Aligned Multilingual Tokenization”
  - Subtitle: “Cross-lingual morpheme-aware vocab + DP decoding”
  - Left panel: “Baseline tokenizer” (your frequency/PMI/length DP without morph signals) showing an imperfect segmentation.
  - Right panel: “Morph-aligned (with UniSeg + CROSS_EQUIV)” showing the same word segmented at morpheme boundaries.
  - Caption under both: “Same word, fewer tokens, grammar preserved.”
- **Example visual (Turkish plural/case):**
  - Word: `Çocuklarımızdan` (“from our children”).
  - Baseline: `Ço · cuk · lar · ımız · dan` (root clipped; tokens: 5).
  - Morph-aligned: `Çocuk · lar · ımız · dan` (root + plural + possessive + ablative; tokens: 4).
  - Highlight “root intact vs clipped” and “+1 token avoided.”
- **Script:** “Tokenization is the first modeling choice. Without morphology, the baseline DP breaks stems and yields inconsistent subwords across languages. Here’s the same Turkish word: baseline fragments the root; the morph-aligned tokenizer keeps morpheme boundaries, producing fewer tokens and preserving grammar. We’ll show how cross-lingual morphology signals drive this behavior.”

## 2) Problem & Stakes
- **Format:** 3 parallel points with icons.
- **Points:** Over-segmentation → longer sequences; boundary violations → lost grammatical cues; inconsistent subwords → poor cross-ling transfer.
- **Script:** “In morph-rich languages one word can encode a whole phrase. Wrong splits inflate context, destroy morphology, and vary by language.”

## 3) Gap in Prior Work
- **Format:** 2-column contrast table.
- **Rows:** Objective (compression vs structure), cross-ling alignment (none vs ad hoc), stability (non-convex/heuristic vs principled).
- **Script:** “BPE/SentencePiece compress, not interpret. Morph-aware variants exist but stay heuristic and lack principled cross-ling alignment.”

## 4) Bit Flip (Big Idea)
- **Format:** Three-pill diagram.
- **Pills:** Convex morphology-regularized embeddings; DP decoder with UniSeg rewards; column-generation vocab growth.
- **Script:** “Flip to a convex embedding objective that encodes morphology, decode with DP that rewards morpheme boundaries, and grow vocab by reduced-cost pricing.”

## 5) Morphology Feature Bank
- **Format:** Layered list; small dictionary icon.
- **Layers:** Char n-grams; affix lexicons; CROSS_EQUIV classes (PLURAL/PROG/etc.); UniSeg boundary hints.
- **Script:** “Each candidate token gets linguistic features. CROSS_EQUIV links functions like English ‘-s’ and Turkish ‘-ler’. UniSeg hints mark plausible boundaries.”

## 6) Cross-Lingual Embedding Objective
- **Format:** Mini formula + before/after cluster sketch.
- **Formula:** Convex GloVe with Laplacian: `||Au-b||^2 + λ tr(Uᵀ L U) + γ||U||²`.
- **Visual:** Plural morphemes clustering across languages.
- **Script:** “Fix feature embeddings, learn token embeddings with a strictly convex loss. Laplacian pulls same-class morphemes together; there is one stable optimum.”

## 7) DP Segmentation with Morphology Rewards
- **Format:** Small lattice or flow arrows.
- **Steps:** Span pruning → DP lattice → costs: NLL + PMI_pen + length ± merge reward + UniSeg boundary rewards + class transitions.
- **Script:** “We score spans with stats, length priors, and morphology rewards. DP picks the min-cost path; UniSeg-aligned spans become cheaper.”

## 8) Column Generation (Vocab Growth)
- **Format:** Loop graphic.
- **Loop:** Decode corpus → price spans by reduced cost → add top-k tokens → refresh embeddings.
- **Script:** “Instead of greedy merges, we add tokens that most reduce segmentation cost. Refresh embeddings so morphology keeps steering the lattice.”

## 9) Evaluation Story
- **Format:** 2-column: metrics + qualitative examples.
- **Metrics:** ByteMush, morph boundary F1, SFI, TPC, GRU PPL (DP+UniSeg vs DP baseline).
- **Qual:** Segmentation examples (Turkish agglutinative; English derivational).
- **Script:** “Boundary F1 rises in morph-rich languages; fragmentation drops; ~20% PPL drop with same TPC. Qualitatively, roots/affixes stay intact.”

## 10) Limits & Coverage
- **Format:** Caution callouts.
- **Points:** Gains largest in morph-rich languages; CROSS_EQUIV/affix coverage sparsity; low-resource gaps.
- **Script:** “Analytic languages see smaller gains. Coverage of equivalence classes is the bottleneck. Low-resource alignment needs better discovery.”

## 11) Future Extensions
- **Format:** Numbered list.
- **Ideas:** (1) Expand CROSS_EQUIV with weak/unsupervised mining; (2) better low-resource transfer; (3) tighter DP–embedding co-training.
- **Script:** “Grow equivalence sets automatically, push to low-resource settings, and co-train DP costs with embedding updates.”

## 12) Closing
- **Format:** Single thesis line + callback visual.
- **Thesis:** “Morphology-regularized, cross-lingual tokenization yields more coherent, transferable tokens without sacrificing efficiency.”
- **Script:** “Tokenization is modeling. Encoding morphology and cross-ling consistency into the objective and decoder makes downstream models’ job easier.”

---

### Optional Slide Variants (swap-ins if time shifts)
- **Ablations (replace slide 10 if needed):** Bar chart: DP baseline vs DP+UniSeg vs No-Affix vs No-UniSeg on morph F1/PPL.
- **Pipeline Summary (replace slide 4 or 8):** End-to-end flow: corpus → candidate spans → feature bank → convex embedding → DP decode → pricing → vocab update.
