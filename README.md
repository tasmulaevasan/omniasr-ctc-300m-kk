# Kazakh ASR — omniASR-CTC-300M fine-tune

Fine-tuning [Meta's Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
`omniASR_CTC_300M_v2` into a Kazakh-specialised speech recognition model, and the full
data → train → eval pipeline to reproduce it.

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-tasmulaev%2Fomniasr--ctc--300m--kk-yellow)](https://huggingface.co/tasmulaev/omniasr-ctc-300m-kk)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

> **Weights and the model card live on Hugging Face:**
> [tasmulaev/omniasr-ctc-300m-kk](https://huggingface.co/tasmulaev/omniasr-ctc-300m-kk).
> This repo holds the *code* to reproduce them; it does not contain the `.pt` file.

## Results

Greedy CTC decoding, no language model. WER = Word Error Rate, CER = character (unit) error
rate. Lower is better.

| Test set  | Base  | Fine-tuned | Δ                     |
| --------- | ----- | ---------- | --------------------- |
| KSC2 (kk) | 40.19 | **18.64**  | **−21.55** (improved) |
| FLEURS en | 17.13 | 58.67      | +41.53 (regressed)    |
| FLEURS ru | 28.91 | 33.52      | +4.61 (regressed)     |

The model more than halves WER on Kazakh while regressing on other languages — the expected
trade-off of monolingual specialisation (catastrophic forgetting). See the
[model card](https://huggingface.co/tasmulaev/omniasr-ctc-300m-kk) for per-clip examples and
full discussion.

## Repository layout

```
.
├── configs/
│   ├── kk-config.yaml        # training config
│   ├── kk-eval.yaml          # eval — fine-tuned checkpoint
│   └── base-eval.yaml        # eval — base model
├── scripts/
│   ├── extract_fleurs.py     # FLEURS .arrow/.wav -> loose .flac + .txt
│   ├── augment.py            # offline waveform augmentation (20% of train)
│   ├── build_parquet.py      # loose files -> sharded parquet dataset
│   ├── build_fleurs_eval.py  # FLEURS en/ru -> eval parquet (forgetting check)
│   ├── run_eval.py           # sweep base vs fine-tuned over all test sets
│   ├── compare_base_ft.py    # qualitative base-vs-fine-tuned transcripts
│   └── check_speaker_overlap.py  # train/test leakage check
├── results/
│   ├── eval_results.csv
│   ├── comparison.md
│   ├── eval_results.csv
│   └── train_results.csv
├── requirements.txt
└── README.md
```

## Setup

Python 3.11. Install a CUDA-matched PyTorch first, then the rest:

```bash
# the model was trained on cu128 / RTX PRO 6000 Blackwell
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`fairseq2` ships a native component (`fairseq2n`) that must match your torch/CUDA build. If
plain `pip` resolution fails, follow the
[omnilingual-asr install docs](https://github.com/facebookresearch/omnilingual-asr).

## Data

Both corpora are public — download them yourself (not redistributed here):

- **ISSAI KSC2** — [issai/Kazakh_Speech_Corpus_2](https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2)
- **Google FLEURS** — [google/fleurs](https://huggingface.co/datasets/google/fleurs) (Kazakh `kk_kz` for training; `en_us` / `ru_ru` for the cross-lingual eval)

Arrange KSC2 as loose `.flac` + `.txt` pairs under `dataset/version=0/corpus=ksc2/{Train,Dev,Test}/`.
FLEURS is converted to the same shape by `extract_fleurs.py`.

## Reproduce

All paths below assume a dataset root of `dataset/version=0`. Adjust to yours.

```bash
# 1. FLEURS (kk) -> loose .flac + .txt, merged into corpus=fleurs/Train/
python scripts/extract_fleurs.py dataset/version=0 --inspect   # peek at the .arrow schema
python scripts/extract_fleurs.py dataset/version=0

# 2. Augment 20% of TRAIN audio in place (noise / time-stretch / room reverb)
python scripts/augment.py dataset/version=0 --dry-run          # preview
python scripts/augment.py dataset/version=0

# 3. Build the sharded parquet dataset (Train + aug -> split=train)
python scripts/build_parquet.py dataset/version=0 parquet_out

# 4. Build FLEURS en/ru eval parquet (for the forgetting check)
python scripts/build_fleurs_eval.py /path/to/fleurs parquet_out
```

Register the dataset and (after training) the model asset card so fairseq2 can resolve them:

```bash
export FAIRSEQ2_USER_ASSET_DIR=$PWD/cards
```

Train (single GPU; tune `grad_accumulation` / `max_num_elements` to your VRAM):

```bash
export OUTPUT_DIR=$PWD/checkpoints
CUDA_VISIBLE_DEVICES=0 python -m workflows.recipes.wav2vec2.asr "$OUTPUT_DIR" \
  --config-file configs/kk-config.yaml
```

Evaluate base vs fine-tuned across KSC2 / FLEURS en / FLEURS ru, collected into one table:

```bash
python scripts/run_eval.py \
  --ft-config configs/kk-eval.yaml \
  --base-config configs/base-eval.yaml \
  --device 0 --out-csv results/eval_results.csv
```

Inspect qualitative differences and verify there is no train/test leakage:

```bash
python scripts/compare_base_ft.py --n 20 --out results/comparison.md
python scripts/check_speaker_overlap.py --audio-hash
```

## Key training settings

Full config in [`configs/kk-config.yaml`](configs/kk-config.yaml). Highlights:

- Base `omniASR_CTC_300M_v2`, tokenizer `omniASR_tokenizer_written_v2`, full fine-tune
- AdamW lr `1.5e-5`, tri-stage schedule `[0.05, 0.35, 0.6]`
- 30,000 steps, effective batch ≈ 36M audio elements/step, bfloat16
- 20% offline waveform augmentation on train; val/test untouched

## Notes & caveats

- **Kazakh only.** Worse than the base model on other languages — not a general multilingual ASR.
- **Greedy decoding.** No LM. A KenLM n-gram + CTC beam search should lower WER further.
- **Normalised output.** Trained on lowercased, punctuation-free text, so output is lowercase with no punctuation.
- **FLEURS kk is in training.** All FLEURS Kazakh splits were folded into train, so FLEURS kk is **not** a valid held-out set for this model. For an independent benchmark use e.g. Common Voice Kazakh test.
- **Splits.** KSC2 splits were prepared locally; a leakage check found 0 byte-identical audio shared between train and test (speaker-level disjointness is unverified — the parquet has no speaker field).

## License

Apache 2.0 — inherited from Omnilingual ASR. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{omnilingualasr2025,
  title  = {Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages},
  author = {{Omnilingual ASR Team}},
  journal = {arXiv preprint arXiv:2511.09690},
  year   = {2025}
}

@inproceedings{mussakhojayeva2022ksc2,
  title     = {KSC2: An Industrial-Scale Open-Source Kazakh Speech Corpus},
  author    = {Mussakhojayeva, Saida and Khassanov, Yerbolat and Varol, Huseyin Atakan},
  booktitle = {Interspeech},
  year      = {2022}
}
```

## Acknowledgements

Built on [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) (Meta) and
the [ISSAI KSC2](https://issai.nu.edu.kz/kz-speech-corpus/) corpus.
