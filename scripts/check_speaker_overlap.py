import argparse
import hashlib
import re
import sys

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
except ImportError:
    sys.exit("pyarrow is required (it ships with fairseq2): pip install pyarrow")

SPEAKER_CANDIDATES = [
    "speaker_id", "speaker", "speakerid", "spk_id", "spk", "spkid",
    "speaker_name", "client_id", "spkr", "spk_idx",
]
ID_CANDIDATES = [
    "utt_id", "uttid", "utterance_id", "id", "audio_id",
    "filename", "file", "path", "audio_path", "wav", "uttkey", "key",
]
AUDIO_CANDIDATES = ["audio_bytes", "audio", "wav", "waveform", "bytes"]
TEXT_CANDIDATES = ["text", "transcript", "transcription", "sentence", "raw_text"]

def pick(colnames, candidates, override=None):
    lower = {c.lower(): c for c in colnames}
    if override:
        if override.lower() in lower:
            return lower[override.lower()]
        sys.exit(f"Column '{override}' not found. Available: {sorted(colnames)}")
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

def to_bytes(a):
    if a is None:
        return None
    if isinstance(a, (bytes, bytearray, memoryview)):
        return bytes(a)
    if isinstance(a, dict):
        b = a.get("bytes")
        if b is not None:
            return to_bytes(b)
        if a.get("path"):
            return str(a["path"]).encode()
        return repr(sorted(a.items())).encode()
    if isinstance(a, str):
        return a.encode()
    if isinstance(a, (list, tuple)):
        try:
            return bytes(a)
        except (ValueError, TypeError):
            return repr(a).encode()
    try:
        return bytes(a)
    except (ValueError, TypeError):
        return repr(a).encode()

def read_values(dataset, corpus, split, col):
    flt = (pc.field("corpus") == corpus) & (pc.field("split") == split)
    return dataset.to_table(columns=[col], filter=flt).column(col).to_pylist()

def report_overlap(label, train_vals, test_vals, regex=None):
    if regex:
        rx = re.compile(regex)

        def key(v):
            m = rx.search(str(v))
            return m.group(1) if m else None

        train_vals = [key(v) for v in train_vals]
        test_vals = [key(v) for v in test_vals]
        if any(v is None for v in test_vals):
            print(f"  [warn] regex did not match every {label} value — check the pattern")

    train_set = {v for v in train_vals if v is not None}
    test_set = {v for v in test_vals if v is not None}
    overlap = train_set & test_set

    print(f"\n=== {label} overlap ===")
    print(f"  distinct in train : {len(train_set):,}")
    print(f"  distinct in test  : {len(test_set):,}")
    print(f"  in BOTH           : {len(overlap):,}")
    if test_set:
        print(f"  share of test {label}s also seen in train: "
              f"{100.0 * len(overlap) / len(test_set):.1f}%")
    if overlap:
        print(f"  examples in both  : {list(overlap)[:5]}")
    return overlap

def _hashes_by_text(dataset, corpus, split, text_col, audio_col, texts):
    flt = ((pc.field("corpus") == corpus) & (pc.field("split") == split)
           & pc.is_in(pc.field(text_col), value_set=pa.array(list(texts))))
    scanner = dataset.scanner(columns=[text_col, audio_col], filter=flt, batch_size=256)
    out = {}
    for batch in scanner.to_batches():
        ts = batch.column(0).to_pylist()
        au = batch.column(1).to_pylist()
        for t, a in zip(ts, au):
            b = to_bytes(a)
            if b is None:
                continue
            out.setdefault(t, set()).add(hashlib.md5(b).hexdigest())
    return out

def targeted_audio_check(dataset, corpus, train_split, test_split,
                         text_col, audio_col, shared_texts):
    print("\n=== audio leak check (restricted to shared transcripts) ===")
    print(f"  comparing audio for {len(shared_texts):,} shared transcripts...")
    train_map = _hashes_by_text(dataset, corpus, train_split, text_col, audio_col, shared_texts)
    test_map = _hashes_by_text(dataset, corpus, test_split, text_col, audio_col, shared_texts)

    leaked, benign = 0, 0
    for t, te_h in test_map.items():
        if te_h & train_map.get(t, set()):
            leaked += 1
        else:
            benign += 1
    print(f"  shared transcripts with BYTE-IDENTICAL audio in train+test (LEAK): {leaked:,}")
    print(f"  shared transcripts that are just re-read by another voice (benign): {benign:,}")
    if test_map:
        print(f"  -> {100.0 * leaked / len(test_map):.1f}% of shared transcripts are true duplicates")
    print("\n  VERDICT (audio):",
          "CLEAN — no identical recordings across splits" if leaked == 0
          else f"LEAK — {leaked:,} test clips are duplicated from train")
    return leaked

def full_audio_check(dataset, corpus, train_split, test_split, audio_col):
    print("\n=== full audio byte-hash check (reads ALL audio — slow) ===")

    def hashes_for(split):
        flt = (pc.field("corpus") == corpus) & (pc.field("split") == split)
        scanner = dataset.scanner(columns=[audio_col], filter=flt, batch_size=256)
        hs, n = set(), 0
        for batch in scanner.to_batches():
            for v in batch.column(0).to_pylist():
                b = to_bytes(v)
                if b is not None:
                    hs.add(hashlib.md5(b).hexdigest()); n += 1
        return hs, n

    test_hashes, n_test = hashes_for(test_split)
    flt = (pc.field("corpus") == corpus) & (pc.field("split") == train_split)
    scanner = dataset.scanner(columns=[audio_col], filter=flt, batch_size=256)
    leaked, n_train = 0, 0
    for batch in scanner.to_batches():
        for v in batch.column(0).to_pylist():
            b = to_bytes(v)
            if b is None:
                continue
            n_train += 1
            if hashlib.md5(b).hexdigest() in test_hashes:
                leaked += 1
    print(f"  hashed {n_test:,} test / scanned {n_train:,} train clips")
    print(f"  byte-identical test clips found in train: {leaked:,}")
    return leaked

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/home/asan_tasmulaev/fine-tune/dataset/version=0")
    ap.add_argument("--corpus", default="ksc2")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--speaker-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--speaker-regex", default=None,
                    help="extract speaker key from the id column, e.g. '^([^_]+)_'")
    ap.add_argument("--audio-hash", action="store_true",
                    help="decisive audio check, restricted to shared transcripts (fast)")
    ap.add_argument("--audio-hash-full", action="store_true",
                    help="exhaustive audio check over the whole train split (slow)")
    args = ap.parse_args()

    try:
        dataset = ds.dataset(args.data, format="parquet", partitioning="hive")
    except Exception as e:
        sys.exit(f"Could not open dataset at {args.data}: {e}")

    cols = dataset.schema.names
    print("Columns in parquet:")
    for c in cols:
        print(f"  - {c}  ({dataset.schema.field(c).type})")

    spk_col = pick(cols, SPEAKER_CANDIDATES, args.speaker_col)
    id_col = pick(cols, ID_CANDIDATES, args.id_col)
    txt_col = pick(cols, TEXT_CANDIDATES)
    audio_col = pick(cols, AUDIO_CANDIDATES)
    print(f"\nDetected -> speaker: {spk_col} | id: {id_col} | text: {txt_col} | audio: {audio_col}")

    if spk_col:
        tr = read_values(dataset, args.corpus, args.train_split, spk_col)
        te = read_values(dataset, args.corpus, args.test_split, spk_col)
        ov = report_overlap("speaker", tr, te)
        print("\n  VERDICT (speaker):",
              "CLEAN (speaker-disjoint)" if not ov else "LEAK: shared speakers")
    else:
        print("\n[!] No speaker column — falling back to transcript + audio checks.")

    if id_col:
        tr = read_values(dataset, args.corpus, args.train_split, id_col)
        te = read_values(dataset, args.corpus, args.test_split, id_col)
        if args.speaker_regex:
            report_overlap("speaker (from id)", tr, te, regex=args.speaker_regex)
        else:
            report_overlap("utterance-id", tr, te)

    shared_texts = set()
    if txt_col:
        tr = read_values(dataset, args.corpus, args.train_split, txt_col)
        te = read_values(dataset, args.corpus, args.test_split, txt_col)
        shared_texts = report_overlap("transcript", tr, te)

    if args.audio_hash:
        if txt_col and shared_texts:
            targeted_audio_check(dataset, args.corpus, args.train_split, args.test_split,
                                 txt_col, audio_col, shared_texts)
        elif audio_col:
            full_audio_check(dataset, args.corpus, args.train_split, args.test_split, audio_col)
    if args.audio_hash_full and audio_col:
        full_audio_check(dataset, args.corpus, args.train_split, args.test_split, audio_col)

    print("\nNote: transcript overlap alone can be benign (shared prompts read by different")
    print("voices). The audio check tells you whether the same RECORDING leaked across splits.")

if __name__ == "__main__":
    main()