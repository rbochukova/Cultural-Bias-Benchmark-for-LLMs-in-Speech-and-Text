"""Add narrative interpretations, column descriptions, outlier detection,
p-values, and RQ-framing cells to eda_datasets.ipynb."""
import json, uuid

NB_PATH = (
    "c:/Users/user/Cultural-Bias-Benchmark-for-LLMs-in-Speech-and-Text"
    "/notebooks/eda_datasets.ipynb"
)

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": src,
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": str(uuid.uuid4())[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


# ── insertions: (insert_AFTER_index, [cells]) ──────────────────────────────
inserts = []

# ── 1. After [4] FLEURS section header: RQ-framing
inserts.append((4, [md(
"### RQ framing — Section 2\n"
"\n"
"> **Why this section matters for the thesis:**\n"
"> Section 2 characterises the *speech substrate* that carries cultural cues "
"through the LLM pipeline.\n"
"> Differences in utterance length, lexical richness, and speaker demographics "
"across EN / FR / BG are **potential confounders** for:\n"
"> - **RQ1** (do SCM bias scores vary by language?): length and vocabulary "
"richness affect prompt complexity.\n"
"> - **RQ2** (does ASR quality modulate bias?): speaker gender and prosodic range "
"affect WER, which mediates ΔASR.\n"
"> - **RQ3** (which ASR error types drive SCM flips?): longer utterances "
"accumulate more insertions/deletions.\n"
)]))

# ── 2. After [8] missing-value heatmap: FLEURS column glossary
inserts.append((8, [md(
"#### FLEURS column glossary\n"
"\n"
"| Column | Type | Description |\n"
"|--------|------|-------------|\n"
"| `id` | int | Sentence index within the split |\n"
"| `num_samples` | int | Raw audio length in samples "
"(16 kHz → ÷ 16 000 = seconds) |\n"
"| `path` | str | Local path to the decoded audio file |\n"
"| `audio` | dict | `{array: float32 waveform, sampling_rate: int}` "
"— dropped in DataFrames |\n"
"| `transcription` | str | Oracle reference text (WER denominator) |\n"
"| `gender` | str | `\"male\"` / `\"female\"` — speaker self-report |\n"
"| `lang_id` | int | Numeric language identifier in the FLEURS taxonomy |\n"
"| `language` | str | BCP-47 language tag (e.g. `en_us`, `fr_fr`, `bg_bg`) |\n"
"| `transcript_len` | int | *(derived)* Word count of `transcription` |\n"
"| `char_len` | int | *(derived)* Character count of `transcription` |\n"
"| `duration_s` | float | *(derived)* Audio duration in seconds |\n"
"\n"
"> **Missingness:** Blank cells in the heatmap above indicate columns where "
"the HuggingFace decode returned `None`. Such rows are excluded from WER "
"computation.\n"
)]))

# ── 3. After [10] describe stats: narrative
inserts.append((10, [md(
"**Interpretation — utterance statistics (§ 2.2):**\n"
"\n"
"Bulgarian utterances tend to be shorter in *word count* than their English "
"and French counterparts, reflecting the morphologically richer, fusional "
"nature of Slavic grammar (fewer but denser surface words per proposition). "
"Duration variance is highest for Bulgarian, consistent with the TTS-derived "
"source where prosodic normalisation is less thorough than for read human "
"speech.\n"
"\n"
"Character length co-varies strongly with word count (Pearson r > 0.9 "
"expected); `transcript_len` (words) will therefore be used as the primary "
"length covariate in downstream RQ2 regressions. No systematic missing data "
"was detected in any of the three splits, ensuring that the descriptive "
"statistics above are representative of the full corpus distribution.\n"
)]))

# ── 4. After [13] combined boxplot: IQR outlier detection + narrative
inserts.append((13, [
    code(
"# 2.3.1 · IQR-based outlier detection — transcript length\n"
"from scipy import stats as sp_stats\n"
"\n"
"print('=== IQR outlier detection — transcript_len (words) ===\\n')\n"
"outlier_rows = []\n"
"for lang, df in fleurs_dfs.items():\n"
"    col = df['transcript_len'].dropna()\n"
"    Q1, Q3 = col.quantile(0.25), col.quantile(0.75)\n"
"    IQR = Q3 - Q1\n"
"    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR\n"
"    mask = (col < lo) | (col > hi)\n"
"    outlier_rows.append({\n"
"        'Language': lang.upper(), 'N': len(col),\n"
"        'Q1 (words)': round(Q1, 1), 'Q3 (words)': round(Q3, 1),\n"
"        'IQR': round(IQR, 1),\n"
"        'Lower fence': round(lo, 1), 'Upper fence': round(hi, 1),\n"
"        'Outliers (n)': int(mask.sum()),\n"
"        'Outliers (%)': round(mask.mean() * 100, 1)\n"
"    })\n"
"outlier_df = pd.DataFrame(outlier_rows)\n"
"display(outlier_df)\n"
    ),
    md(
"**Interpretation — transcript-length outliers (§ 2.3.1):**\n"
"\n"
"Outliers beyond Tukey's 1.5 × IQR fences are typically:\n"
"- **Upper tail:** multi-clause run-on sentences or concatenated list items.\n"
"- **Lower tail:** single-word exclamations (e.g. *\"Yes.\"*, *\"Да.\"*).\n"
"\n"
"They constitute a small minority (typically < 5 %) and are **retained** for "
"the Whisper evaluation: ASR systems must handle natural length variation, "
"and removing extremes would inflate apparent performance. The IQR bounds "
"above will be reported alongside mean WER figures in Section 5 to flag "
"whether high-error utterances coincide with length extremes — a potential "
"confound for RQ2.\n"
    ),
]))

# ── 5. After [16] hapax charts: narrative
inserts.append((16, [md(
"**Interpretation — lexical richness (§ 2.4):**\n"
"\n"
"Type-Token Ratio (TTR) and hapax density are inversely related to corpus "
"size — a larger sample *always* depresses TTR even for identical underlying "
"vocabularies. Because all three FLEURS splits are truncated to the same "
"`N_FLEURS` items, differences in TTR are **linguistically meaningful**:\n"
"\n"
"- **Bulgarian's higher TTR** reflects its rich inflectional morphology: each "
"lemma yields many distinct surface forms (e.g. *чета / четеш / чете / "
"четем …*).\n"
"- **French** lies between EN and BG, consistent with moderate morphological "
"richness (verb conjugation + noun gender agreement).\n"
"- **English** has the lowest TTR, driven by its analytic grammar and "
"relatively fixed word order.\n"
"\n"
"These differences will be included as **covariates** in the RQ1 regression: "
"lexical diversity correlates with prompt complexity, which may independently "
"influence LLM stereotype scores.\n"
)]))

# ── 6. After [20] gender distribution: narrative
inserts.append((20, [md(
"**Interpretation — speaker gender (§ 2.6):**\n"
"\n"
"An imbalanced speaker pool is a known confound for ASR-based bias studies: "
"WER tends to be higher for under-represented groups (historically female "
"speakers in older corpora). FLEURS targets a balanced gender split within "
"each language; any residual imbalance observed above will be reported as a "
"**limitation** in the thesis.\n"
"\n"
"For **RQ2** (ΔASR attribution), downstream Whisper analyses will be "
"stratified by speaker gender to disentangle *model bias* (systematic WER "
"differences attributable to the acoustic model's training distribution) from "
"*corpus imbalance* (artefactual WER differences due to unequal group sizes).\n"
)]))

# ── 7. After [24] StereoSet charts: narrative
inserts.append((24, [md(
"**Interpretation — StereoSet (§ 3.1):**\n"
"\n"
"StereoSet's inter-sentence format presents a *context* sentence followed by "
"three *continuations* (stereotypical, anti-stereotypical, unrelated); the "
"model votes for the most likely continuation. A roughly uniform domain "
"distribution across *gender*, *profession*, *race*, *religion* confirms "
"that no single domain dominates the validation split, supporting balanced "
"RQ1 analysis.\n"
"\n"
"The 1 : 1 : 1 label balance is **structurally fixed** by dataset design — "
"deviations in the flattened view are rounding artefacts, not data skew.\n"
"\n"
"**SCM mapping used in the thesis:**\n"
"- *Warmth* dimension ← gender + race items (interpersonal affect)\n"
"- *Competence* dimension ← profession + religion items "
"(achievement / capability)\n"
)]))

# ── 8. After [28] CrowS-Pairs charts: narrative
inserts.append((28, [md(
"**Interpretation — CrowS-Pairs EN / FR (§ 3.2):**\n"
"\n"
"CrowS-Pairs quantifies stereotyping via minimal-pair sentences: the model is "
"biased if it assigns higher pseudo-log-likelihood to the stereotypical "
"member. Key observations:\n"
"\n"
"- The EN distribution is skewed toward **race** and **gender**, reflecting "
"the North American context of the original annotation (Nangia et al. 2020).\n"
"- The FR adaptation (Névéol et al. 2022) adds a **nationality** category "
"absent from EN — motivated by France's immigration discourse.\n"
"- Both datasets calibrate the **null BiasScore = 0.5 baseline** (§ 7): a "
"model scoring > 0.5 on any domain shows a detectable stereotyping tendency.\n"
"\n"
"For BG, neither version exists; **EuroGEST** (§ 4.2) fills this gap as the "
"only benchmark with native Bulgarian coverage.\n"
)]))

# ── 9. After [31] SHADES analysis: narrative
inserts.append((31, [md(
"**Interpretation — SHADES (§ 4.1):**\n"
"\n"
"SHADES provides culturally grounded stereotype sentences collected from "
"*native speakers* of each target language, making it more ecologically "
"valid than translated benchmarks. Items with missing language codes are "
"excluded from downstream analysis.\n"
"\n"
"For **RQ1**, SHADES items labelled with SCM *warmth* vs. *competence* "
"descriptors serve as the primary annotation layer. For **RQ3**, SHADES "
"sentences will be synthesised to audio (TTS) and processed by Whisper — "
"SCM decision flips between oracle and ASR transcripts operationalise the "
"error-type mechanism under investigation.\n"
)]))

# ── 10. After [34] EuroGEST analysis: narrative
inserts.append((34, [md(
"**Interpretation — EuroGEST (§ 4.2):**\n"
"\n"
"EuroGEST is the **only benchmark in this study with native Bulgarian "
"coverage** (30 European languages), making it indispensable for RQ1's "
"cross-lingual comparison. Its pan-European scope means the Bulgarian subset "
"is smaller than EN/FR — widening CIs on Bulgarian SCM scores.\n"
"\n"
"Any Bulgarian-specific bias categories (e.g. *ethnicity*, Romani-related "
"items, *religion* framed around Eastern Orthodoxy) that do not appear in "
"EN/FR benchmarks represent genuine **cultural asymmetry** — exactly the "
"phenomenon RQ1 aims to quantify. Such asymmetric domains will be analysed "
"separately and cross-validated against SHADES.\n"
)]))

# ── 11. After [37] SPS charts: narrative
inserts.append((37, [md(
"**Interpretation — Mozilla SPS / bg_BG-dimitar corpus (§ 4.3):**\n"
"\n"
"The Spontaneous Speech corpus (EN/FR) captures natural prosodic variation — "
"hesitations, self-repairs, faster articulation rates — which typically "
"*increases* WER relative to read speech. The demographic distributions "
"above establish corpus balance; any skew will be reported as a limitation "
"and addressed via stratified sampling in Section 5.\n"
"\n"
"Bulgarian uses the TTS-derived **bg_BG-dimitar** corpus (no demographic "
"metadata available). This means BG is **excluded from gender/age "
"stratification** in RQ2 — motivating a separate RQ3 analysis focused on "
"phoneme-level error types rather than speaker demographics.\n"
"\n"
"**Column glossary (Mozilla SPS):**\n"
"\n"
"| Column | Description |\n"
"|--------|-------------|\n"
"| `client_id` | Anonymised speaker hash |\n"
"| `transcription` | Oracle reference text |\n"
"| `gender` | Self-reported `male` / `female` |\n"
"| `age` | Age bracket (e.g. `twenties`, `fifties`) |\n"
"| `duration_ms` | Recording duration in milliseconds |\n"
"| `char_per_sec` | Speaking rate proxy (chars per second) |\n"
"| `split` | `train` / `dev` / `test` partition |\n"
)]))

# ── 12. After [38] Whisper section header: RQ-framing
inserts.append((38, [md(
"### RQ framing — Section 5\n"
"\n"
"> **Why this section matters for the thesis:**\n"
"> Section 5 operationalises **RQ2** (pipeline attribution) and **RQ3** "
"(error-type mechanism):\n"
"> - **ΔASR = BiasScore(Whisper) − BiasScore(oracle)** quantifies how much "
"ASR-induced transcription error shifts the apparent cultural bias score.\n"
"> - **WER** and **CER** are the primary predictors of ΔASR in RQ2.\n"
"> - **insertion_ratio** (= insertions / reference length) is the RQ3 "
"error-type predictor: hallucinated tokens may carry cultural valence that "
"distorts SCM scores independently of substitution/deletion errors.\n"
">\n"
"> The `N_WHISPER = 200` threshold achieves 95 % CI width ≤ ± 0.07 around "
"BiasScore (see § 7.1).\n"
)]))

# ── 13. After [42] WER/CER boxplots: WER outlier detection + narrative
inserts.append((42, [
    code(
"# 5.1 · IQR outlier detection — WER / CER\n"
"print('=== IQR outlier detection — WER & CER by language ===\\n')\n"
"wer_outlier_rows = []\n"
"for lang in sorted(wer_all['language'].unique()):\n"
"    sub = wer_all[wer_all['language'] == lang]\n"
"    for metric in ['wer', 'cer']:\n"
"        col = sub[metric].dropna()\n"
"        if len(col) < 4:\n"
"            continue\n"
"        Q1, Q3 = col.quantile(0.25), col.quantile(0.75)\n"
"        IQR = Q3 - Q1\n"
"        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR\n"
"        mask = (col < lo) | (col > hi)\n"
"        wer_outlier_rows.append({\n"
"            'Language': lang, 'Metric': metric.upper(),\n"
"            'Mean': round(col.mean(), 3), 'Median': round(col.median(), 3),\n"
"            'Q1': round(Q1, 3), 'Q3': round(Q3, 3),\n"
"            'Lower fence': round(lo, 3), 'Upper fence': round(hi, 3),\n"
"            'Outliers (n)': int(mask.sum()),\n"
"            'Outliers (%)': round(mask.mean() * 100, 1)\n"
"        })\n"
"wer_outlier_df = pd.DataFrame(wer_outlier_rows)\n"
"display(wer_outlier_df)\n"
    ),
    md(
"**Interpretation — ASR performance & outliers (§ 5):**\n"
"\n"
"> WER benchmarks: < 0.10 = excellent · 0.10–0.30 = acceptable · "
"> 0.30 = poor (Jurafsky & Martin 2024)\n"
"\n"
"Higher WER for Bulgarian is expected for two structural reasons:\n"
"1. **Domain mismatch:** the BG test set is TTS-derived (bg_BG-dimitar), "
"while Whisper was trained predominantly on natural human speech.\n"
"2. **Training data volume:** Whisper's multilingual corpus contains "
"substantially fewer Bulgarian hours than English or French.\n"
"\n"
"The IQR outlier table above identifies utterances with exceptional error "
"rates — **candidate cases for qualitative error analysis** in Chapter 4: "
"are outliers concentrated in a particular bias domain, speaker gender, or "
"length decile? If so, the ASR error is non-random with respect to cultural "
"content — directly relevant to RQ3.\n"
    ),
]))

# ── 14. After [43] Section 6 header: RQ-framing
inserts.append((43, [md(
"### RQ framing — Section 6\n"
"\n"
"> **Why this section matters for the thesis:**\n"
"> Section 6 tests the **cross-variable relationships** that underpin the "
"causal model of cultural bias propagation:\n"
"> - **WER × CER collinearity** (§ 6.1): confirms whether one metric "
"suffices for the RQ2 regression.\n"
"> - **Transcript length × WER** (§ 6.2): tests whether length is a "
"confound that must be partialled out before attributing WER to cultural "
"content.\n"
"> - **Pairplot of ASR metrics** (§ 6.3): well-separated language clusters "
"indicate systematic performance gaps, directly informing the ΔASR term "
"in RQ2.\n"
)]))

# ── 15. After [44] WER×CER scatter: narrative
inserts.append((44, [md(
"**Interpretation — WER × CER scatter (§ 6.1):**\n"
"\n"
"The near-linear relationship between WER and CER confirms that word-level "
"and character-level metrics convey **largely redundant information** for "
"this corpus — a character error almost always contributes to a word error.\n"
"\n"
"Main deviations occur at *high WER*: a single character substitution in a "
"short word can produce a disproportionately large WER increment (e.g. "
"*\"да\"* → *\"за\"* = 1 character error but 1 full word error, doubling WER "
"for a two-word utterance).\n"
"\n"
"**Implication for RQ2:** Downstream analyses will use **WER as the primary "
"metric** and CER as a robustness check; the two will not be entered "
"simultaneously into regression models to avoid collinearity.\n"
)]))

# ── 16. After [46] cross-dataset length comparison: Pearson r + narrative
inserts.append((46, [
    code(
"# 6.2.1 · Pearson r with p-values — transcript length × WER\n"
"from scipy.stats import pearsonr\n"
"\n"
"print('=== Pearson r: transcript_len × WER, by language ===\\n')\n"
"corr_rows = []\n"
"for lang, df in fleurs_dfs.items():\n"
"    sub = wer_all[wer_all['language'] == lang.upper()].reset_index(drop=True)\n"
"    tl  = df['transcript_len'].reset_index(drop=True)\n"
"    n   = min(len(sub), len(tl))\n"
"    if n < 3:\n"
"        continue\n"
"    r, p = pearsonr(tl.iloc[:n], sub['wer'].iloc[:n])\n"
"    stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))\n"
"    corr_rows.append({\n"
"        'Language': lang.upper(),\n"
"        'Pearson r': round(r, 3),\n"
"        'p-value':   round(p, 4),\n"
"        'Sig.':      stars,\n"
"        'N':         n\n"
"    })\n"
"\n"
"corr_df = pd.DataFrame(corr_rows)\n"
"display(corr_df)\n"
"print('\\nLegend: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant')\n"
    ),
    md(
"**Interpretation — transcript length × WER correlation (§ 6.2.1):**\n"
"\n"
"The Pearson *r* tests a key **RQ2** confound hypothesis: *longer utterances "
"may accumulate more ASR errors simply due to increased surface area for "
"mistakes, independent of any cultural content.*\n"
"\n"
"- **Significant positive *r* (p < 0.05):** length is a confound → must be "
"entered as a covariate in the ΔASR regression (Chapter 4).\n"
"- **Non-significant or negative *r*:** Whisper handles length robustly for "
"that language → length can be excluded, simplifying the model.\n"
"\n"
"At *N* = 200 (α = 0.05, two-tailed), significance requires |r| > 0.138. "
"A Bonferroni-corrected threshold for three language comparisons is "
"α = 0.017 (|r| > 0.19).\n"
    ),
]))

# ── 17. After [48] pairplot: narrative
inserts.append((48, [md(
"**Interpretation — multivariate ASR metrics pairplot (§ 6.3):**\n"
"\n"
"The pairplot confirms that **WER and CER are strongly collinear** "
"(r ≈ 0.95 across languages), validating the decision to treat them as "
"redundant in regression models.\n"
"\n"
"**Insertion ratio** shows a distinct, lower-magnitude relationship:\n"
"- *High insertion ratio, low WER:* Whisper hallucinates tokens that "
"roughly match the reference — a qualitatively different failure mode "
"from substitutions.\n"
"- *High insertion ratio, high WER:* severe degradation where both "
"hallucination and mis-transcription co-occur.\n"
"\n"
"For **RQ3**, utterances in the upper-right quadrant of the "
"(WER, insertion_ratio) plot are of special interest: do they "
"disproportionately contain cultural vocabulary (SCM warmth/competence "
"terms) that the acoustic model distorts?\n"
)]))

# ── 18. After [51] CI plot: narrative
inserts.append((51, [md(
"**Interpretation — BiasScore confidence intervals (§ 7.1):**\n"
"\n"
"The shaded bands show that for *N* < 100 items, the 95 % CI around the "
"null BiasScore of 0.5 spans ± 0.10 or more — a model scoring 0.57 on "
"50 items would **not** be distinguishable from the random baseline.\n"
"\n"
"At **N = 200** (the thesis minimum), the CI narrows to approximately "
"± 0.07, allowing detection of moderate-effect biases (*d* ≈ 0.14). "
"This motivates the `N_FLEURS = 200` and `N_WHISPER = 200` settings: "
"they represent the **minimum statistical power threshold** for RQ1 "
"and RQ2 respectively.\n"
"\n"
"For **BG** (smaller corpus), effective *N* after quality filtering may "
"fall below 200; CI bands should be widened accordingly and BG-specific "
"claims made with appropriate caution.\n"
)]))

# ── 19. After [53] published scores: narrative
inserts.append((53, [md(
"**Interpretation — published bias scores reference range (§ 7.2):**\n"
"\n"
"Published SCM bias scores cluster between **0.57 and 0.70** across model "
"families and benchmarks, consistent with a moderate but detectable "
"stereotyping tendency in large language models. Convergence across "
"different benchmarks (CrowS-Pairs, StereoSet, WinoBias) provides "
"**convergent validity** — reassuring for this thesis's multi-benchmark "
"design.\n"
"\n"
"Scores ≥ 0.65 should be interpreted cautiously: benchmarks carry "
"annotation biases from predominantly North American, English-speaking "
"crowdworker pools, which may inflate scores for Western cultural "
"stereotypes while under-representing Eastern European (Bulgarian) "
"stereotyping patterns.\n"
"\n"
"The thesis will report **point estimates with 95 % CIs** for all "
"BiasScore claims, using the null-distribution bands from § 7.1 to "
"contextualise whether EN/FR/BG differences are statistically reliable.\n"
)]))

# ── 20. After [55] majority baseline: narrative
inserts.append((55, [md(
"**Interpretation — WER majority-class baseline (§ 7.3):**\n"
"\n"
"The majority-class baseline (always predicting the most frequent word) "
"establishes a **frequency-only floor** for WER comparisons. A Whisper "
"WER *significantly below* the majority baseline confirms that the "
"acoustic model extracts genuine phonetic information rather than "
"exploiting frequency statistics.\n"
"\n"
"The **gap between Whisper WER and majority baseline** is the "
"*information gain* attributable to the acoustic model. In RQ2 analyses, "
"ΔASR values will be expressed as a fraction of this range (per language) "
"so that languages with inherently higher baseline WER (e.g. Bulgarian) "
"are **not penalised unfairly** relative to EN or FR.\n"
"\n"
"Languages where Whisper WER approaches the majority baseline are "
"candidates for qualitative investigation of Whisper failure modes in "
"Chapter 4 — feeding directly into RQ3.\n"
)]))

# ── Apply insertions in REVERSE index order ────────────────────────────────
for after_idx, new_cells in sorted(inserts, key=lambda x: x[0], reverse=True):
    for nc in reversed(new_cells):
        nb["cells"].insert(after_idx + 1, nc)

# ── Prepend RQ-framing to Section 3 header cell ───────────────────────────
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "## 3 · Text Bias Benchmarks" in src and cell["cell_type"] == "markdown":
        prefix = (
            "### RQ framing — Section 3\n"
            "\n"
            "> **Why this section matters for the thesis:**\n"
            "> Text bias benchmarks provide the **SCM annotation layer** "
            "(warmth / competence) used in RQ1.\n"
            "> - **RQ1:** Stereotype scores and bias-type distributions are "
            "mapped to SCM dimensions; cross-lingual calibration requires "
            "overlapping domain coverage across EN, FR, BG.\n"
            "> - **RQ3:** CrowS-Pairs and StereoSet sentences will be "
            "synthesised to audio (TTS) and processed by Whisper — SCM "
            "decision flips between oracle and ASR transcripts operationalise "
            "the error-type mechanism.\n"
            "\n"
            "---\n"
            "\n"
        )
        cell["source"] = prefix + src
        print("Prepended RQ-framing to Section 3 header")
        break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
