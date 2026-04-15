# Cultural Bias Benchmark for LLMs in Speech and Text

A multilingual benchmark that measures cultural bias in large language models (LLMs) under two conditions — standard text prompts and a speech pipeline where audio is first transcribed by Whisper before being scored by the LLM.

The benchmark covers **English, French, and Bulgarian** and probes biases along the Stereotype Content Model (SCM) dimensions of **warmth** and **competence**. Three research questions are investigated:

- **RQ1 (Cultural grounding):** Do bias scores differ by language, SCM dimension, and item origin (parallel-translated vs. culture-specific native)?
- **RQ2 (Pipeline attribution):** How large is the ASR-attributable modality gap (ΔASR) when comparing Whisper-to-LLM against oracle-transcript-to-LLM?
- **RQ3 (Error-type mechanism):** Which ASR error types — negation changes, deletion-heavy segments, trait-cue substitutions — most strongly predict SCM decision flips, beyond raw WER?

---

## Repository Structure

```
├── data/
│   ├── stimuli_seed.csv           # 6,893 validated SCM forced-choice probes (EN/FR/BG)
│   ├── results/
│   │   ├── text/                  # LLM inference results (natural / grammar / typical variants)
│   │   └── speech/                # Whisper → LLM inference results (natural / grammar / typical)
│   └── audio/                     # TTS-generated audio clips (not tracked in git)
├── src/
│   ├── inference_text.py          # Score stimuli with a text LLM
│   ├── inference_speech.py        # TTS → Whisper → LLM pipeline
│   ├── tts.py                     # Text-to-speech via OpenAI TTS
│   ├── asr.py                     # Whisper ASR transcription
│   ├── score.py                   # BiasScore + RQ1/RQ2 statistical analyses
│   ├── visualize.py               # All thesis figures
│   ├── rq3_error_types.py         # RQ3 logistic regression (error-type mechanism)
│   └── add_*.py                   # Dataset curation scripts
├── reports/figures/               # Generated figures (PNG)
├── notebooks/
│   └── eda_datasets.ipynb         # Exploratory data analysis
├── tests/
│   └── test_stimulus.py           # Stimulus validation tests
├── requirements.txt               # Project dependencies
└── requirements_frozen.txt        # Fully pinned environment snapshot
```

---

## Installation

Requires **Python 3.10**.

```bash
git clone https://github.com/rbochukova/Cultural-Bias-Benchmark-for-LLMs-in-Speech-and-Text
cd Cultural-Bias-Benchmark-for-LLMs-in-Speech-and-Text
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

---

## Usage

### Text condition (RQ1)

```bash
# Score all stimuli with GPT-4o-mini (natural prompt variant)
python src/inference_text.py --model gpt-4o-mini

# Run statistical analyses and print results table
python src/score.py

# Generate all figures
python src/visualize.py --no-show
```

### Speech condition (RQ2)

```bash
# Generate audio with TTS
python src/tts.py

# Transcribe with Whisper large-v3, then score with LLM
python src/inference_speech.py --asr-model large-v3 --llm-model gpt-4o-mini

# Analyses including ΔASR are printed by score.py automatically
# when speech results are present
python src/score.py
```

### Prompt variants

Both scripts accept `--prompt-variant grammar` and `--prompt-variant typical` to replicate the robustness checks.

### RQ3 error-type analysis

```bash
python src/rq3_error_types.py
```

Outputs a logistic regression table (`flip ~ WER + error_type_features + lang + dim`) and saves a two-panel forest plot to `reports/figures/rq3_logreg.png`.

---

## Key Results (GPT-4o-mini, Whisper large-v3)

| Condition | BiasScore | vs. null (0.50) |
|---|---|---|
| Text – overall | 0.442 | below null (anti-stereotypical lean) |
| Speech – overall | 0.452 | below null |
| ΔASR (overall) | +0.010 | small positive modality gap |

- **FR/warmth** is the only language × dimension cell with a significant bias score after FDR correction (negative / anti-stereotypical direction).
- **ΔASR** is consistent but small across prompt variants (natural: +0.010, grammar: +0.021, typical: −0.002).
- **RQ3:** deletion-heavy segments (OR ≈ 4.5) and negation flips (OR ≈ 3.0) are the strongest predictors of SCM decision flips beyond WER alone (LR test χ²(5) = 19.1, p = .002).

---

## Acknowledgements

Stimuli are derived from or inspired by:
- [StereoSet](https://github.com/moinnadeem/StereoSet) (Nadeem et al., 2021)
- [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) (Nangia et al., 2020)
- [French CrowS-Pairs](https://github.com/pixelastic/crows-pairs) (Névéol et al., 2022)
- [SHADES](https://github.com/antndlcrx/shades) (de la Croix, 2024)
- [WinoBias](https://github.com/uclanlp/corefBias) (Zhao et al., 2018)

ASR: [OpenAI Whisper](https://github.com/openai/whisper) (Radford et al., 2023).
LLM scoring: [OpenAI API](https://platform.openai.com/).

MSc thesis project — Information Studies (Data Science track), University of Amsterdam, 2025.