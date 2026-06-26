import argparse
import io
import sys
import tempfile

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    import soundfile as sf
    import numpy as np
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Need pyarrow, soundfile, numpy.")

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return 100.0 * edit_distance(r, h) / max(len(r), 1)

def cer(ref, hyp):
    return 100.0 * edit_distance(list(ref), list(hyp)) / max(len(ref), 1)

def to_raw(audio_list):
    return bytes((x & 0xFF) for x in audio_list)

def decode(raw, diag=False):
    try:
        data, srate = sf.read(io.BytesIO(raw), dtype="float32")
        if diag:
            print(f"  [diag] decoded as audio container, sr={srate}, samples={len(data)}")
        return data, srate
    except Exception:
        pcm = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if diag:
            print(f"  [diag] not a container; treated as int16 PCM @16k, samples={len(pcm)}")
        return pcm, 16000

def sample_rows(data_dir, corpus, split, n):
    dataset = ds.dataset(data_dir, format="parquet", partitioning="hive")
    flt = (pc.field("corpus") == corpus) & (pc.field("split") == split)
    tbl = dataset.to_table(columns=["text", "audio_bytes", "audio_size"], filter=flt)
    total = tbl.num_rows
    idx = list(range(total))
    step = max(total // n, 1)
    idx = idx[::step][:n]
    texts = tbl.column("text").to_pylist()
    auds = tbl.column("audio_bytes").to_pylist()
    sizes = tbl.column("audio_size").to_pylist()
    if idx:
        print(f"  [diag] first clip: len(audio_bytes)={len(auds[idx[0]])}, "
              f"audio_size={sizes[idx[0]]}")
    return [(texts[i], auds[i]) for i in idx]

def transcribe_all(card, wavs, lang, device):
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    pipe = ASRInferencePipeline(model_card=card, device=device)
    return pipe.transcribe(wavs, lang=[lang] * len(wavs), batch_size=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/home/asan_tasmulaev/fine-tune/dataset/version=0")
    ap.add_argument("--corpus", default="ksc2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--card-base", default="omniASR_CTC_300M_v2")
    ap.add_argument("--card-ft", default="omniASR_CTC_300M_kk",
                    help="name field of your fine-tuned user asset card")
    ap.add_argument("--lang", default="kaz_Cyrl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="comparison.md")
    args = ap.parse_args()

    print(f"Sampling {args.n} clips from {args.corpus}/{args.split}...")
    rows = sample_rows(args.data, args.corpus, args.split, args.n)

    wav_paths, refs = [], []
    tmp = tempfile.mkdtemp(prefix="cmp_")
    for k, (text, aud) in enumerate(rows):
        data, srate = decode(to_raw(aud), diag=(k == 0))
        path = f"{tmp}/clip_{k:03d}.wav"
        sf.write(path, data, srate)
        wav_paths.append(path)
        refs.append(text)

    print("Transcribing with BASE model...")
    base = transcribe_all(args.card_base, wav_paths, args.lang, args.device)
    print("Transcribing with FINE-TUNED model...")
    ft = transcribe_all(args.card_ft, wav_paths, args.lang, args.device)

    items = []
    for ref, b, f in zip(refs, base, ft):
        items.append({
            "ref": ref, "base": b, "ft": f,
            "base_cer": cer(ref, b), "ft_cer": cer(ref, f),
            "improve": cer(ref, b) - cer(ref, f),
        })
    items.sort(key=lambda x: x["improve"], reverse=True)

    lines = []
    lines.append("### Qualitative comparison (KSC2 test)\n")
    lines.append("Sample transcriptions, base `omniASR_CTC_300M_v2` vs this fine-tune. "
                 "Sorted by how much fine-tuning reduced character error on each clip.\n")
    lines.append("| Reference | Base | Fine-tuned |")
    lines.append("|-----------|------|------------|")
    for it in items:
        r = it["ref"].replace("|", "\\|")
        b = it["base"].replace("|", "\\|")
        f = it["ft"].replace("|", "\\|")
        lines.append(f"| {r} | {b} | {f} |")
    md = "\n".join(lines) + "\n"

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    print(f"\nSaved markdown to {args.out}")
    mean_b = sum(i["base_cer"] for i in items) / len(items)
    mean_f = sum(i["ft_cer"] for i in items) / len(items)
    print(f"Sample mean CER: base {mean_b:.1f} -> fine-tuned {mean_f:.1f}")

if __name__ == "__main__":
    main()