import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import (
    AddGaussianNoise,
    Compose,
    RoomSimulator,
    TimeStretch,
)

TARGET_SR     = 16_000
AUGMENT_RATIO = 0.20
SEED          = 42

TRAIN_CORPORA = ["corpus=ksc2", "corpus=fleurs"]

def make_augmenter():
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=False, p=0.5),
        RoomSimulator(p=0.5),
    ])

def load_mono(path: Path):
    with sf.SoundFile(str(path)) as f:
        wav = f.read(dtype="float32")
        sr = f.samplerate
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav, sr

def save_flac(wav: np.ndarray, sr: int, path: Path):
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = wav / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, format="FLAC")

def augment_corpus(corpus_root: Path, ratio: float, dry_run: bool):
    train_dir = corpus_root / "Train"
    aug_dir   = corpus_root / "aug"
    name = corpus_root.name

    if not train_dir.exists():
        print(f"  [{name}] Train/ not found, skipping")
        return

    flacs = sorted(train_dir.rglob("*.flac"))
    if not flacs:
        print(f"  [{name}] no .flac in Train/, skipping")
        return

    random.seed(SEED)
    k = int(len(flacs) * ratio)
    selected = random.sample(flacs, k=k)
    print(f"  [{name}] {len(flacs):,} files, augmenting {k:,} ({ratio*100:.0f}%)")

    augmenter = make_augmenter()
    done = 0
    for orig in selected:
        rel = orig.relative_to(train_dir)
        out_flac = aug_dir / rel
        txt_orig = orig.with_suffix(".txt")

        if dry_run:
            done += 1
            continue

        wav, sr = load_mono(orig)
        wav = augmenter(wav, sample_rate=sr)
        save_flac(wav, sr, out_flac)

        if txt_orig.exists():
            out_txt = out_flac.with_suffix(".txt")
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt_orig, out_txt)
            txt_orig.unlink()
        orig.unlink()

        done += 1
        if done % 500 == 0:
            print(f"    ... {done:,}/{k:,}")

    print(f"  [{name}] {'(dry-run) would augment' if dry_run else 'augmented'} "
          f"{done:,} -> {aug_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path, help="fine-tune/dataset/version=0/")
    ap.add_argument("--ratio", type=float, default=AUGMENT_RATIO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("DRY RUN -- nothing written or deleted\n")

    random.seed(SEED)
    np.random.seed(SEED)

    for corpus in TRAIN_CORPORA:
        root = args.dataset_root / corpus
        if root.exists():
            augment_corpus(root, args.ratio, args.dry_run)
        else:
            print(f"  {corpus} not found, skipping")

    print("\nDone" if not args.dry_run else "\nDry run complete.")

if __name__ == "__main__":
    main()