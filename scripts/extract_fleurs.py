import argparse
import io
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000
SPLITS = ["train", "dev", "test"]

def to_16k_mono(waveform: np.ndarray, sr: int) -> np.ndarray:
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
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
    return waveform

def save_pair(waveform, sr, out_flac: Path, text: str):
    out_flac.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_flac), to_16k_mono(waveform, sr), TARGET_SR, format="FLAC")
    out_flac.with_suffix(".txt").write_text(text.strip().lower(), encoding="utf-8")

def decode_audio_field(audio):
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
        if audio.get("bytes") is not None:
            with sf.SoundFile(io.BytesIO(audio["bytes"])) as f:
                return f.read(dtype="float32"), f.samplerate
        if audio.get("path"):
            with sf.SoundFile(audio["path"]) as f:
                return f.read(dtype="float32"), f.samplerate
    raise ValueError(f"unrecognized audio field: {type(audio)}")

def pick_text_column(columns):
    for c in ["transcription", "raw_transcription", "normalized_text", "text", "sentence"]:
        if c in columns:
            return c
    raise ValueError(f"no text column found among {columns}")

def read_arrow_shard(arrow_path: Path):
    try:
        from datasets import Dataset
        ds = Dataset.from_file(str(arrow_path))
        text_col = pick_text_column(ds.column_names)
        audio_col = "audio" if "audio" in ds.column_names else None
        for row in ds:
            if audio_col:
                wav, sr = decode_audio_field(row[audio_col])
            else:
                raise ValueError(f"no 'audio' column in {ds.column_names}")
            yield wav, sr, str(row[text_col])
        return
    except Exception as e:
        print(f"    datasets reader failed ({e}); trying raw pyarrow")

    import pyarrow as pa
    with pa.memory_map(str(arrow_path)) as src:
        try:
            reader = pa.ipc.open_stream(src)
        except Exception:
            src.seek(0)
            reader = pa.ipc.open_file(src)
        table = reader.read_all()

    cols = table.column_names
    text_col = pick_text_column(cols)
    audio_col = "audio" if "audio" in cols else None
    if audio_col is None:
        raise ValueError(f"no 'audio' column in arrow schema: {cols}")

    audio_arr = table.column(audio_col).to_pylist()
    text_arr = table.column(text_col).to_pylist()
    for audio, text in zip(audio_arr, text_arr):
        wav, sr = decode_audio_field(audio)
        yield wav, sr, str(text)

def extract_split(fleurs_dir: Path, split: str, out_train_dir: Path):
    data_dir = fleurs_dir / "data" / "kk_kz"
    tar_path = data_dir / "audio" / f"{split}.tar.gz"
    tsv_path = data_dir / f"{split}.tsv"
    if not tar_path.exists():
        print(f"  [{split}] {tar_path.name} not found, skipping")
        return 0

    tmp_dir = data_dir / "audio" / f"_extract_{split}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{split}] extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    arrow_files = sorted(tmp_dir.rglob("*.arrow"))
    wav_files   = sorted(tmp_dir.rglob("*.wav"))
    out_sub = out_train_dir / f"fleurs_{split}"
    count = 0

    if arrow_files:
        print(f"  [{split}] found {len(arrow_files)} .arrow shard(s)")
        for shard in arrow_files:
            for wav, sr, text in read_arrow_shard(shard):
                save_pair(wav, sr, out_sub / f"{split}_{count:07d}.flac", text)
                count += 1
                if count % 500 == 0:
                    print(f"    ... {count:,} samples")
    elif wav_files:
        print(f"  [{split}] found {len(wav_files)} .wav files; reading text from {tsv_path.name}")
        manifest = {}
        if tsv_path.exists():
            for line in tsv_path.read_text(encoding="utf-8").splitlines():
                p = line.split("\t")
                if len(p) >= 4:
                    manifest[p[1].strip()] = p[3].strip()
        by_name = {w.name: w for w in wav_files}
        for filename, text in manifest.items():
            wpath = by_name.get(filename)
            if wpath is None:
                continue
            with sf.SoundFile(str(wpath)) as f:
                wav = f.read(dtype="float32", always_2d=False)
                sr = f.samplerate
            save_pair(wav, sr, out_sub / filename.replace(".wav", ".flac"), text)
            count += 1
    else:
        print(f"  [{split}] WARNING: no .arrow or .wav found inside the tar")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  [{split}] -> {count:,} flac+txt written to {out_sub}")
    return count

def inspect(fleurs_dir: Path):
    data_dir = fleurs_dir / "data" / "kk_kz"
    tar_path = data_dir / "audio" / "train.tar.gz"
    tmp = data_dir / "audio" / "_inspect"
    tmp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members[:10]:
            tar.extract(m, tmp)
    arrow = sorted(tmp.rglob("*.arrow"))
    wav = sorted(tmp.rglob("*.wav"))
    print(f"Inside {tar_path.name}: {len(arrow)} .arrow (in first members), {len(wav)} .wav")
    if arrow:
        try:
            from datasets import Dataset
            ds = Dataset.from_file(str(arrow[0]))
            print("Columns:", ds.column_names)
            print("Features:", ds.features)
            print("First row keys:", list(ds[0].keys()))
            if "audio" in ds.column_names:
                a = ds[0]["audio"]
                print("Audio field keys:", list(a.keys()) if isinstance(a, dict) else type(a))
        except Exception as e:
            print("datasets inspect failed:", e)
    shutil.rmtree(tmp, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path, help="fine-tune/dataset/version=0/")
    ap.add_argument("--inspect", action="store_true",
                    help="only print the .arrow schema, do not extract everything")
    args = ap.parse_args()

    fleurs_dir = args.dataset_root / "corpus=fleurs"
    if not fleurs_dir.exists():
        print(f"corpus=fleurs not found at {fleurs_dir}")
        return

    if args.inspect:
        inspect(fleurs_dir)
        return

    out_train = fleurs_dir / "Train"
    total = 0
    for split in SPLITS:
        total += extract_split(fleurs_dir, split, out_train)
    print(f"\nOK FLEURS extracted: {total:,} samples -> {out_train}")
    print("All FLEURS splits merged into Train/ (variant A).")

if __name__ == "__main__":
    main()