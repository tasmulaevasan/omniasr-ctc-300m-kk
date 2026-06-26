import argparse
import io
import re
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

LANGUAGE        = "kaz_Cyrl"
TARGET_SR       = 16_000
ROWS_PER_FILE   = 4_000
ROW_GROUP_SIZE  = 100

SCHEMA = pa.schema([
    pa.field("text",        pa.string()),
    pa.field("audio_bytes", pa.list_(pa.int8())),
    pa.field("audio_size",  pa.int64()),
    pa.field("corpus",      pa.dictionary(pa.int32(), pa.string())),
    pa.field("split",       pa.dictionary(pa.int32(), pa.string())),
    pa.field("language",    pa.dictionary(pa.int32(), pa.string())),
])

LAYOUT = {
    "ksc2": {
        "train": ["Train", "aug"],
        "dev":   ["Dev"],
        "test":  ["Test"],
    },
    "fleurs": {
        "train": ["Train", "aug"],
    },
}

def load_audio_bytes(path: Path):
    with sf.SoundFile(str(path)) as f:
        waveform = f.read(dtype="float32")
        sr = f.samplerate

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if sr != TARGET_SR:
        try:
            import resampy
            waveform = resampy.resample(waveform, sr, TARGET_SR)
        except ImportError:
            n = int(len(waveform) * TARGET_SR / sr)
            waveform = np.interp(np.linspace(0, len(waveform) - 1, n),
                                 np.arange(len(waveform)), waveform).astype(np.float32)

    peak = np.abs(waveform).max()
    if peak > 1.0:
        waveform = waveform / peak

    buf = io.BytesIO()
    sf.write(buf, waveform, TARGET_SR, format="FLAC")
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8).astype(np.int8)
    return arr.tolist(), len(waveform)

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def gather_pairs(dirs):
    pairs = []
    for d in dirs:
        if not d.exists():
            continue
        for flac in sorted(d.rglob("*.flac")):
            txt = flac.with_suffix(".txt")
            if txt.exists():
                pairs.append((flac, txt))
    return pairs

def _write_part(buffer, out_dir, part_idx, corpus, split):
    table = pa.table(
        {
            "text":        pa.array([r["text"] for r in buffer],        type=pa.string()),
            "audio_bytes": pa.array([r["audio_bytes"] for r in buffer], type=pa.list_(pa.int8())),
            "audio_size":  pa.array([r["audio_size"] for r in buffer],  type=pa.int64()),
            "corpus":      pa.array([corpus] * len(buffer),   type=pa.dictionary(pa.int32(), pa.string())),
            "split":       pa.array([split] * len(buffer),    type=pa.dictionary(pa.int32(), pa.string())),
            "language":    pa.array([LANGUAGE] * len(buffer), type=pa.dictionary(pa.int32(), pa.string())),
        },
        schema=SCHEMA,
    )
    out_path = out_dir / f"part-{part_idx}.parquet"
    pq.write_table(table, str(out_path), row_group_size=ROW_GROUP_SIZE)

def build_split(pairs, out_dir, corpus, split):
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer, part_idx, total, skipped = [], 0, 0, 0

    for flac_path, txt_path in pairs:
        text = normalize_text(txt_path.read_text(encoding="utf-8"))
        if not text:
            skipped += 1
            continue
        try:
            audio_bytes, audio_size = load_audio_bytes(flac_path)
        except Exception as e:
            print(f"    WARNING: failed {flac_path.name}: {e}")
            skipped += 1
            continue

        buffer.append({"text": text, "audio_bytes": audio_bytes, "audio_size": audio_size})
        total += 1

        if len(buffer) >= ROWS_PER_FILE:
            _write_part(buffer, out_dir, part_idx, corpus, split)
            part_idx += 1
            buffer = []
            if part_idx % 20 == 0:
                print(f"    ... {total:,} rows written ({part_idx} parts)")

    if buffer:
        _write_part(buffer, out_dir, part_idx, corpus, split)
        part_idx += 1

    extra = f" ({skipped} skipped)" if skipped else ""
    print(f"  OK {corpus}/{split}: {total:,} rows in {part_idx} part files{extra}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_root",  type=Path, help="fine-tune/dataset/version=0/")
    ap.add_argument("output_root", type=Path, help="output dir")
    args = ap.parse_args()

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    for corpus, splits in LAYOUT.items():
        corpus_root = args.input_root / f"corpus={corpus}"
        if not corpus_root.exists():
            print(f"skip corpus={corpus}: {corpus_root} not found")
            continue
        print(f"\n=== corpus={corpus} ===")
        for split, folders in splits.items():
            dirs = [corpus_root / f for f in folders]
            pairs = gather_pairs(dirs)
            print(f"  [{split}] {len(pairs):,} flac+txt pairs from {folders}")
            if not pairs:
                continue
            out_dir = out / f"corpus={corpus}" / f"split={split}" / f"language={LANGUAGE}"
            build_split(pairs, out_dir, corpus, split)

    print("\nDone")

if __name__ == "__main__":
    main()