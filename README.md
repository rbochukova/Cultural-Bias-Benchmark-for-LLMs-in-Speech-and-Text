# Cultural Bias Benchmark for LLMs in Speech and Text

A multilingual benchmark that measures cultural bias in large language models (LLMs) under two conditions — standard text prompts and a speech pipeline where audio is first transcribed by Whisper before being scored by the LLM.

The benchmark covers **English, French, and Bulgarian** and probes biases along the Stereotype Content Model (SCM) dimensions of **warmth** and **competence**. Three research questions are investigated:

- **RQ1 (Cultural grounding):** Do bias scores differ by language, SCM dimension, and item origin (parallel-translated vs. culture-specific native)?
- **RQ2 (Pipeline attribution):** How large is the ASR-attributable modality gap (ΔASR) when comparing Whisper-to-LLM against oracle-transcript-to-LLM?
- **RQ3 (Error-type mechanism):** Which ASR error types — negation changes, deletion-heavy segments, trait-cue substitutions — most strongly predict SCM decision flips, beyond raw WER?

## Stimuli

Stimuli are forced-choice sentence pairs: one stereotypical, one anti-stereotypical. Each item is assigned to either the **warmth** or **competence** dimension of the Stereotype Content Model. Dimension labels were assigned manually by the author. Items were drawn from existing bias datasets (see Acknowledgements) and supplemented with original sentences authored from scratch. All items were manually reviewed and validated before use.

Items are categorised by origin:
- **Parallel** - the same underlying social scenario appears in two or more languages (cross-language aligned groups, prefixed `PG-` for gender, `PN-` for nationality)
- **Native** - culture-specific items with no cross-language counterpart


## Repository Structure

```
├── data/
│   ├── stimuli_seed.csv           # Validated SCM forced-choice probes (EN/FR/BG)
│   ├── results/
│   │   ├── text/                  # LLM inference results (natural / grammar / typical variants)
│   │   └── speech/                # Whisper → LLM inference results
│   └── audio/                     # TTS-generated audio clips (not tracked in git)
├── src/
│   ├── inference_text.py          # Score stimuli via OpenAI logprobs (text condition)
│   ├── inference_speech.py        # TTS → Whisper → LLM pipeline (speech condition)
│   ├── inference_mdeberta.py      # mDeBERTa-v3-base PLL scoring (cross-encoder baseline)
│   ├── tts.py                     # Text-to-speech via Azure Cognitive Services
│   ├── asr.py                     # Whisper ASR transcription + WER/CER computation
│   ├── score.py                   # BiasScore + RQ1/RQ2 statistical analyses
│   ├── visualize.py               # All thesis figures
│   ├── rq3_error_types.py         # RQ3 logistic regression (error-type mechanism)
│   ├── fidelity_check.py          # Stimulus fidelity checking (cue-word coverage)
│   ├── validate_csv.py            # CSV schema validation
│   ├── stimulus_builder.py        # Initial stimulus CSV seeding
│   ├── stimulus_expander.py       # Expands CSV with CrowS-Pairs and EuroGEST items
│   ├── add_crowspairs_items.py    # Ingests EN/FR CrowS-Pairs items
│   ├── add_stereoset_profession.py# Ingests EN StereoSet profession items
│   └── add_winobias_items.py      # Ingests EN WinoBias gender-profession items
├── link_parallel_items.py         # Assigns PG-/PN- parallel group IDs across languages
├── reports/figures/               # Generated figures (PNG)
├── notebooks/
│   └── eda_datasets.ipynb         # Exploratory data analysis
├── tests/
│   └── test_stimulus.py           # Stimulus validation tests
├── requirements.txt               # Project dependencies
└── requirements_frozen.txt        # Fully pinned environment snapshot
```


## Installation

Requires **Python 3.10**.

```bash
git clone https://github.com/rbochukova/Cultural-Bias-Benchmark-for-LLMs-in-Speech-and-Text
cd Cultural-Bias-Benchmark-for-LLMs-in-Speech-and-Text
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your API keys:

```
OPENAI_API_KEY=sk-...
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=westeurope
```

`AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` are only required for TTS audio generation (`tts.py`).


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
# Generate audio with Azure TTS
python src/tts.py

# Transcribe with Whisper large-v3, then score with LLM
python src/inference_speech.py --asr-model large-v3 --llm-model gpt-4o-mini

# Analyses including ΔASR are printed by score.py automatically
# when speech results are present
python src/score.py
```

### Prompt variants

Both inference scripts accept `--prompt-variant grammar` and `--prompt-variant typical` to replicate the robustness checks.

### RQ3 error-type analysis

```bash
python src/rq3_error_types.py
```

Outputs a logistic regression table (`flip ~ WER + error_type_features + lang + dim`) and saves a two-panel forest plot to `reports/figures/rq3_logreg.png`.


## Acknowledgements

Stimuli are derived from or inspired by:
- [StereoSet](https://github.com/moinnadeem/StereoSet) (Nadeem et al., 2021)
- [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) (Nangia et al., 2020)
- [French CrowS-Pairs](https://github.com/pixelastic/crows-pairs) (Névéol et al., 2022)
- [SHADES](https://github.com/antndlcrx/shades) (de la Croix, 2024)
- [WinoBias](https://github.com/uclanlp/corefBias) (Zhao et al., 2018)
- [EuroGEST](https://huggingface.co/datasets/utter-project/EuroGEST) (Utter Project)

ASR: [OpenAI Whisper](https://github.com/openai/whisper) (Radford et al., 2023).
LLM scoring: [OpenAI API](https://platform.openai.com/).
TTS: [Azure Cognitive Services Speech](https://azure.microsoft.com/en-us/products/ai-services/ai-speech).

MSc thesis project — Information Studies (Data Science track), University of Amsterdam, 2026.
