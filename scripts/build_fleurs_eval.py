import argparse
import io
import re
import shutil
import tarfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

TARGET_SR      = 16_000
ROWS_PER_FILE  = 4000
ROW_GROUP_SIZE = 100
SPLITS         = ["train", "dev", "test"]

LANG_MAP   = {"en_us": "eng_Latn", "ru_ru": "rus_Cyrl"}
CORPUS_MAP = {"en_us": "fleurs_en", "ru_ru": "fleurs_ru"}

SCHEMA = pa.schema([
    pa.field("text",        pa.string()),
    pa.field("audio_bytes", pa.list_(pa.int8())),
    pa.field("audio_size",  pa.int64()),
])

def load_audio_bytes(path: Path):
    with sf.SoundFile(str(path)) as f:
        wav = f.read(dtype="float32")
        sr = f.samplerate
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        try:
            import resampy
            wav = resampy.resample(wav, sr, TARGET_SR)
        except ImportError:
            n = int(len(wav) * TARGET_SR / sr)
            wav = np.interp(np.linspace(0, len(wav) - 1, n),
                            np.arange(len(wav)), wav).astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = wav / peak
    buf = io.BytesIO()
    sf.write(buf, wav, TARGET_SR, format="FLAC")
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8).astype(np.int8)
    return arr.tolist(), len(wav)

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip())

def write_part(buffer, out_dir, part_idx):
    table = pa.table(
        {
            "text":        pa.array([r["text"] for r in buffer],        type=pa.string()),
            "audio_bytes": pa.array([r["audio_bytes"] for r in buffer], type=pa.list_(pa.int8())),
            "audio_size":  pa.array([r["audio_size"] for r in buffer],  type=pa.int64()),
        },
        schema=SCHEMA,
    )
    pq.write_table(table, str(out_dir / f"part-{part_idx}.parquet"), row_group_size=ROW_GROUP_SIZE)

def build_split(pairs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer, part_idx, total, skipped = [], 0, 0, 0
    for wav_path, text in pairs:
        t = normalize_text(text)
        if not t:
            skipped += 1
            continue
        try:
            ab, asize = load_audio_bytes(wav_path)
        except Exception as e:
            print(f"    skip {wav_path.name}: {e}")
            skipped += 1
            continue
        buffer.append({"text": t, "audio_bytes": ab, "audio_size": asize})
        total += 1
        if len(buffer) >= ROWS_PER_FILE:
            write_part(buffer, out_dir, part_idx)
            part_idx += 1
            buffer = []
    if buffer:
        write_part(buffer, out_dir, part_idx)
        part_idx += 1
    extra = f" ({skipped} skipped)" if skipped else ""
    print(f"  OK {total} rows in {part_idx} parts -> {out_dir}{extra}")

def process_lang(fleurs_root: Path, lang: str, out_root: Path):
    corpus   = CORPUS_MAP[lang]
    language = LANG_MAP[lang]
    data_dir = fleurs_root / lang
    print(f"\n=== {lang} -> corpus={corpus}, language={language} ===")

    pairs, temp_dirs = [], []
    for split in SPLITS:
        tar_path = data_dir / "audio" / f"{split}.tar.gz"
        tsv_path = data_dir / f"{split}.tsv"
        if not tar_path.exists():
            print(f"  [{split}] {tar_path.name} not found, skipping")
            continue
        tmp = data_dir / "audio" / f"_extract_{split}"
        tmp.mkdir(parents=True, exist_ok=True)
        temp_dirs.append(tmp)
        print(f"  [{split}] extracting {tar_path.name} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp)

        manifest = {}
        if tsv_path.exists():
            for line in tsv_path.read_text(encoding="utf-8").splitlines():
                p = line.split("\t")
                if len(p) >= 4:
                    manifest[p[1].strip()] = p[3].strip()
        by_name = {w.name: w for w in tmp.rglob("*.wav")}
        n_before = len(pairs)
        for fname, text in manifest.items():
            wp = by_name.get(fname)
            if wp is not None:
                pairs.append((wp, text))
        print(f"  [{split}] {len(pairs) - n_before} clips")

    out_dir = out_root / "version=0" / f"corpus={corpus}" / "split=test" / f"language={language}"
    build_split(pairs, out_dir)

    for t in temp_dirs:
        shutil.rmtree(t, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fleurs_root", type=Path, help="dir containing en_us/ ru_ru/ subdirs")
    ap.add_argument("out_root",    type=Path, help="parquet_out (the dir holding version=0)")
    ap.add_argument("--langs", nargs="*", default=["en_us", "ru_ru"])
    args = ap.parse_args()

    for lang in args.langs:
        if lang not in LANG_MAP:
            print(f"unknown lang {lang}, skipping")
            continue
        if not (args.fleurs_root / lang).exists():
            print(f"{args.fleurs_root / lang} not found, skipping")
            continue
        process_lang(args.fleurs_root, lang, args.out_root)

    print("\nDone")

if __name__ == "__main__":
    main()