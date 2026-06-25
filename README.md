# Cultural Bias Benchmark for LLMs in Speech and Text

A multilingual benchmark that measures cultural bias in large language models (LLMs) under two conditions: standard **text** prompts and a **speech** pipeline where audio is first transcribed by an ASR system before being scored by the LLM.

The benchmark covers **English, French, and Bulgarian** and probes biases along the Stereotype Content Model (SCM) dimensions of **warmth** and **competence**. Three research questions are investigated:

- **Cultural grounding:** Do bias scores differ by language, SCM dimension, and item origin (parallel-translated vs. culture-specific native)?
- **Pipeline attribution:** How large is the ASR-attributable modality gap (ΔASR) between the Whisper/Azure-to-LLM pipeline and the oracle-transcript-to-LLM baseline?
- **Error-type mechanism:** Which ASR error types: negation changes, deletion-heavy segments, trait-cue substitutions - most strongly predict SCM decision flips, beyond raw WER?

## Models and conditions

Six models spanning aligned (instruction-tuned/RLHF) and unaligned families are scored with a forced-choice protocol (the model picks the more "natural" of a stereotypical vs. anti-stereotypical sentence):

| Model | Scoring | Inference script |
|---|---|---|
| GPT-4o-mini | A/B token log-probs (OpenAI API) | `src/inference_text.py` |
| Llama-3.2-3B-Instruct | A/B token log-probs (chat template, local) | `src/inference_instruct_lm.py` |
| mDeBERTa-v3-base | Pseudo-log-likelihood (masked-LM) | `src/inference_mdeberta.py` |
| BLOOM-7B1 | Mean per-token log-prob (causal LM) | `src/inference_causal_lm.py` |
| Mistral-7B-v0.1 | Mean per-token log-prob (causal LM) | `src/inference_causal_lm.py` |
| Llama-3.2-3B (base) | Mean per-token log-prob (causal LM) | `src/inference_causal_lm.py` |

`src/inference_vllm.py` provides an OpenAI-compatible vLLM client for running the large local models on a SLURM cluster.

The speech condition is evaluated under **four ASR systems**: Whisper `large-v3`, `medium`, `small` (`src/asr.py`) and Azure Speech-to-Text (`src/asr_azure.py`), so ΔASR can be traced to transcription quality (WER ≈ 4.9–16.0%).

## Stimuli

Stimuli are forced-choice sentence pairs: one stereotypical, one anti-stereotypical. Each item is assigned to either the **warmth** or **competence** dimension of the SCM. Dimension labels were assigned manually by the author; inter-annotator agreement on a stratified subsample is reported in the thesis (see `src/iaa_*.py`). Items were drawn from existing bias datasets (see Acknowledgements) and supplemented with original sentences. All items were manually reviewed and validated before use.

Items are categorised by origin:
- **Parallel**: the same underlying social scenario appears across languages (cross-language aligned groups, prefixed `PG-` for gender, `PN-` for nationality).
- **Native**: culture-specific items with no cross-language counterpart.

## Repository structure

```
├── data/
│   ├── stimuli_seed.csv               # Validated SCM forced-choice probes (EN/FR/BG)
│   ├── parallel_fidelity.csv          # Back-translation fidelity scores for parallel pairs
│   ├── iaa_coding_sheet.csv           # Blind 2nd-annotator coding sheet (IAA)
│   ├── iaa_key.csv                    # Gold key for the IAA subsample
│   ├── results/
│   │   ├── text/                      # LLM text-condition results (per model + prompt variants)
│   │   ├── speech/                    # <asr>_<model>_results.csv (ASR → LLM)
│   │   └── asr/                       # <asr>_transcripts.csv with per-item WER/CER
│   └── audio/                         # TTS-generated audio clips (not tracked in git)
├── src/
│   ├── stimulus_builder.py            # Seed the stimulus CSV from source benchmarks
│   ├── stimulus_expander.py           # Expand with CrowS-Pairs + EuroGEST items
│   ├── add_crowspairs_items.py        # Ingest EN/FR CrowS-Pairs items
│   ├── add_stereoset_profession.py    # Ingest EN StereoSet profession items
│   ├── add_winobias_items.py          # Ingest EN WinoBias gender-profession items
│   ├── link_parallel_items.py         # Assign PG-/PN- parallel group IDs across languages
│   ├── validate_csv.py                # Stimulus CSV schema validation
│   ├── fidelity_check.py              # Parallel-pair fidelity (cue coverage + back-translation)
│   ├── inference_text.py              # GPT-4o-mini scoring via OpenAI log-probs
│   ├── inference_instruct_lm.py       # Llama-Instruct scoring via chat template (local)
│   ├── inference_causal_lm.py         # BLOOM / Mistral / Llama-base log-prob scoring
│   ├── inference_mdeberta.py          # mDeBERTa-v3-base pseudo-log-likelihood scoring
│   ├── inference_vllm.py              # vLLM client for cluster runs
│   ├── inference_speech.py            # Score ASR transcripts (speech condition)
│   ├── asr.py                         # Whisper transcription + per-item WER/CER
│   ├── asr_azure.py                   # Azure Speech-to-Text transcription
│   ├── tts.py                         # Azure Text-to-Speech audio generation
│   ├── score.py                       # BiasScore + RQ1/RQ2 statistics
│   ├── rq3_error_types.py             # RQ3 logistic regression (error-type mechanism)
│   ├── make_figures.py                # Main thesis figures (Fig 1–3)
│   ├── appendix_figures.py            # Appendix figures A1–A9
│   ├── iaa_sample.py                  # Draw stratified IAA subsample + blind coding sheet
│   ├── iaa_kappa.py                   # Cohen's kappa with bootstrap CI
│   ├── reviewer_response.py           # Profession-exclusion, GEE, tokenisation checks
│   ├── translit_control.py            # Cyrillic→Latin script control (BG tokenisation)
│   └── robustness_extra.py            # Surface-controlled origin effect, source, prompt variants
├── figures/                          # Generated figures (PNG)
├── notebooks/eda_datasets.ipynb       # Exploratory data analysis
├── tests/test_stimulus.py             # Stimulus validation tests
├── jobs/run_bias_inference.job        # SLURM job: vLLM server + cluster inference
├── requirements.txt                   # Project dependencies
└── requirements_frozen.txt            # Fully pinned environment snapshot
```

## Installation

Requires **Python 3.10**.

```bash
git clone https://github.com/rbochukova/Cultural-Bias-Benchmark-for-LLMs-in-Speech-and-Text
cd Cultural-Bias-Benchmark-for-LLMs-in-Speech-and-Text
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=sk-...
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=westeurope
HF_TOKEN=hf_...
```

`AZURE_SPEECH_*` are required for TTS (`tts.py`) and Azure ASR (`asr_azure.py`); `HF_TOKEN` for the gated HuggingFace models and EuroGEST.

## Usage

### Text condition (Cultural grounding)

```bash
# Score with each model
python src/inference_text.py     --model gpt-4o-mini
python src/inference_mdeberta.py
python src/inference_causal_lm.py --model mistralai/Mistral-7B-v0.1
python src/inference_instruct_lm.py --model meta-llama/Llama-3.2-3B-Instruct

# Statistics and results tables
python src/score.py
```

### Speech condition (Pipeline attribution)

```bash
# 1. Generate audio (Azure TTS)
python src/tts.py

# 2. Transcribe (per ASR system)
python src/asr.py --model large-v3        # also: medium, small
python src/asr_azure.py

# 3. Score the transcripts
python src/inference_speech.py     --asr-model large-v3 --llm-model gpt-4o-mini
python src/inference_causal_lm.py  --asr-model large-v3 --model mistralai/Mistral-7B-v0.1

# ΔASR attribution is printed by score.py when speech results are present
python src/score.py
```

### Prompt variants

The text inference scripts accept `--prompt-variant grammar` and `--prompt-variant typical` for the robustness checks.

### Error-type analysis

```bash
python src/rq3_error_types.py
```

Fits `flip ~ WER + error_type_features + lang + dim` and saves a forest plot to `figures/`.

### Figures and robustness

```bash
python src/make_figures.py             # Fig 1–3
python src/appendix_figures.py         # Appendix A1–A9
python src/reviewer_response.py        # Profession / GEE / tokenisation checks
python src/translit_control.py         # Cyrillic→Latin script control (BG nationality items)
python src/robustness_extra.py         # Surface-controlled + source + prompt-variant checks
python src/iaa_kappa.py                # Inter-annotator agreement
```

## Acknowledgements

Stimuli are derived from or inspired by:
- [StereoSet](https://github.com/moinnadeem/StereoSet) (Nadeem et al., 2021)
- [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) (Nangia et al., 2020)
- [French CrowS-Pairs](https://gitlab.inria.fr/french-crows-pairs) (Névéol et al., 2022)
- [SHADES](https://huggingface.co/datasets/LanguageShades/BiasShades) (Mitchell et al., 2025)
- [WinoBias](https://github.com/uclanlp/corefBias) (Zhao et al., 2018)
- [EuroGEST](https://huggingface.co/datasets/utter-project/EuroGEST) (Utter Project)

ASR: [OpenAI Whisper](https://github.com/openai/whisper) (Radford et al., 2023) and [Azure Speech-to-Text](https://azure.microsoft.com/en-us/products/ai-services/ai-speech).
LLM scoring: [OpenAI API](https://platform.openai.com/) and [HuggingFace Transformers](https://huggingface.co/docs/transformers).
TTS: [Azure Cognitive Services Speech](https://azure.microsoft.com/en-us/products/ai-services/ai-speech).

MSc thesis project - Information Studies (Data Science track), University of Amsterdam, 2026.
</content>
</invoke>
