"""Restructure eda_datasets.ipynb to match the UvA EDA template format.
Run on Windows: python scripts/restructure_template.py

Template sections:
  1 · Personal Information
  2 · Data Context  (1-2 paragraphs + methodology diagram)
  3 · Data Description
      3.1 · FLEURS
      3.2 · Text Bias Benchmarks
      3.3 · Gated Datasets
      3.4 · Whisper ASR — WER / CER Analysis
      3.5 · Multivariate Analysis
      3.6 · Baseline Model & Bias Scores
  4 · Consolidated Schema Summary
  5 · Provenance Log
  6 · References
"""
import json, uuid, sys
sys.stdout.reconfigure(encoding="utf-8")

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


cells = nb["cells"]

# ── 1. Update Cell [0]: Personal Information ─────────────────────────────
cells[0]["source"] = (
    "# 1 · Personal Information\n"
    "\n"
    "| | |\n"
    "|---|---|\n"
    "| **Name** | Ralitsa Petkova Bochukova |\n"
    "| **Student ID** | *(UvA ID)* |\n"
    "| **Email** | *(student@student.uva.nl)* |\n"
    "| **Programme** | MSc Information Studies — Data Science |\n"
    "| **Course** | Thesis EDA |\n"
    "| **Submitted on** | DD.MM.YYYY |\n"
    "\n"
    "**Thesis:** *Measuring Cultural Bias in Multilingual LLMs Across Text and Speech: "
    "A Warmth-Competence Benchmark with ASR Attribution*\n"
)

# ── 2. Update Cell [1]: Data Context (was "## 1 · Research Questions…") ──
# Find it by content
for i, cell in enumerate(cells):
    if "## 1" in "".join(cell["source"]) and "Research Questions" in "".join(cell["source"]):
        data_context_idx = i
        break

cells[data_context_idx]["source"] = (
    "## 2 · Data Context\n"
    "\n"
    "This notebook supports the thesis *Measuring Cultural Bias in Multilingual "
    "LLMs Across Text and Speech: A Warmth-Competence Benchmark with ASR "
    "Attribution* (UvA, MSc Information Studies — Data Science, 2026). "
    "The study investigates whether large language models reproduce "
    "Stereotype Content Model (SCM) warmth/competence biases differently "
    "across English, French, and Bulgarian, and whether Automatic Speech "
    "Recognition (ASR) errors amplify or suppress those biases when "
    "cultural stimuli are delivered as speech rather than text.\n"
    "\n"
    "Six complementary datasets are combined: **FLEURS** (multilingual read "
    "speech, primary audio carrier), **StereoSet** and **CrowS-Pairs** "
    "(EN/FR stereotype benchmarks), **SHADES** and **EuroGEST** (gated "
    "multilingual bias datasets including Bulgarian), and the **Mozilla "
    "Spontaneous Speech corpus** / **bg_BG-dimitar** TTS corpus (speaker "
    "demographic metadata for ASR stratification). Together they provide "
    "the text, audio, and demographic layers required to address all three "
    "research questions.\n"
    "\n"
    "### Research Questions\n"
    "\n"
    "| RQ | Question |\n"
    "|:---|:---------|\n"
    "| **RQ1** — Cultural grounding | How do SCM bias scores vary by language, "
    "SCM dimension (warmth vs. competence), and item origin (parallel-translated "
    "vs. culture-specific native)? |\n"
    "| **RQ2** — Pipeline attribution | How large is ΔASR = BiasScore(Whisper) − "
    "BiasScore(oracle), and how does it correlate with WER/CER? |\n"
    "| **RQ3** — Error-type mechanism | Which ASR error types most strongly "
    "predict SCM decision flips beyond WER? |\n"
    "\n"
    "**Languages:** English (EN), French (FR), Bulgarian (BG). "
    "BG is lower-resource and treated as a sensitivity variable.\n"
    "\n"
    "### Dataset overview\n"
    "\n"
    "| Dataset | Role | Source | Languages |\n"
    "|---------|------|--------|-----------|\n"
    "| FLEURS | Primary speech corpus | HuggingFace `google/fleurs` | EN, FR, BG |\n"
    "| StereoSet | Text bias (intrasentence) | HuggingFace `McGill-NLP/stereoset` | EN |\n"
    "| CrowS-Pairs | Text bias (minimal pairs) | HuggingFace `crows_pairs` (EN) / `Hobbit1069/French_CrowS-Pairs` (FR) | EN, FR |\n"
    "| SHADES | Multilingual stereotype cloze | HuggingFace (gated) | EN, FR, + others |\n"
    "| EuroGEST | European stereotype benchmark | HuggingFace (gated) `utter-project/EuroGEST` | EN, FR, **BG** + 27 more |\n"
    "| Mozilla SPS | Spontaneous speech + demographics | Local tar.gz | EN, FR |\n"
    "| bg_BG-dimitar | Bulgarian TTS corpus | Local tar.gz | BG |\n"
    "\n"
    "### Data aggregation pipeline\n"
    "\n"
    "![Data aggregation pipeline](methodology_diagram.png)\n"
    "\n"
    "> *Save your thesis data pipeline diagram as `notebooks/methodology_diagram.png` "
    "to render the image above.*\n"
)

# ── 3. Insert "## 3 · Data Description" before the FLEURS section ─────────
# Find the FLEURS section (## 2 · FLEURS)
fleurs_header_idx = None
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    # Must START with the header line (not just contain FLEURS in body text)
    if cell["cell_type"] == "markdown" and src.startswith("## 2") and "FLEURS" in src.split("\n")[0]:
        fleurs_header_idx = i
        break

if fleurs_header_idx is not None:
    data_desc_cell = md(
        "## 3 · Data Description\n"
        "\n"
        "The following sections present the exploratory data analysis for each "
        "dataset. For each corpus the analysis covers: schema overview, "
        "descriptive statistics, univariate distributions, outlier detection "
        "(IQR method), and—where datasets overlap—multivariate cross-dataset "
        "comparisons. Significance tests use α = 0.05 (Bonferroni-corrected "
        "for multiple comparisons within a section).\n"
    )
    cells.insert(fleurs_header_idx, data_desc_cell)
    print(f"Inserted '## 3 · Data Description' before cell [{fleurs_header_idx}]")
    # Adjust index for subsequent operations
    fleurs_header_idx += 1

# ── 4. Renumber section headers ───────────────────────────────────────────
renames = {
    "## 2 · FLEURS": "### 3.1 · FLEURS",
    "## 3 · Text Bias Benchmarks": "### 3.2 · Text Bias Benchmarks",
    "## 4 · Gated Datasets": "### 3.3 · Gated Datasets",
    "## 5 · Whisper ASR": "### 3.4 · Whisper ASR",
    "## 6 · Multivariate Analysis": "### 3.5 · Multivariate Analysis",
    "## 7 · Baseline Model": "### 3.6 · Baseline Model & Bias Scores",
    "## 8 · Consolidated Schema Summary": "## 4 · Consolidated Schema Summary",
    "## 9 · Provenance Log": "## 5 · Provenance Log",
    "## 10 · References": "## 6 · References",
}

for cell in cells:
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])
    for old, new in renames.items():
        if old in src:
            cell["source"] = src.replace(old, new, 1)
            print(f"Renamed: '{old}' → '{new}'")
            break

# ── 5. Update internal RQ-framing labels to match new section numbers ─────
rq_renames = {
    "### RQ framing — Section 2": "### RQ framing — § 3.1 (FLEURS)",
    "### RQ framing — Section 3": "### RQ framing — § 3.2 (Text Bias Benchmarks)",
    "### RQ framing — Section 5": "### RQ framing — § 3.4 (Whisper ASR)",
    "### RQ framing — Section 6": "### RQ framing — § 3.5 (Multivariate Analysis)",
}

for cell in cells:
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])
    for old, new in rq_renames.items():
        if old in src:
            cell["source"] = src.replace(old, new, 1)
            print(f"Updated RQ label: '{old}' → '{new}'")
            break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone. Notebook now has {len(cells)} cells.")

# Print final structure
print("\nFinal section headers:")
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    if cell["cell_type"] == "markdown" and src.startswith("#"):
        first_line = src.split("\n")[0]
        if first_line.startswith("#"):
            print(f"  [{i:02d}] {first_line[:80]}")
