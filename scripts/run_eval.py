import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

CORPORA = [
    ("ksc2",      "KSC2 (kaz)"),
    ("fleurs_en", "FLEURS en"),
    ("fleurs_ru", "FLEURS ru"),
]

WER_PATTERNS = [
    re.compile(r'Word Error Rate \(WER\):\s*([\d.]+)'),
    re.compile(r'Word Error Rate \(WER\)\s*\|\s*Last:\s*([\d.]+)'),
]
UER_PATTERNS = [
    re.compile(r'Unit Error Rate \(UER\):\s*([\d.]+)'),
    re.compile(r'Unit Error Rate \(UER\)\s*\|\s*Last:\s*([\d.]+)'),
]

def set_partition_filter(text: str, corpus: str) -> str:
    lines = text.splitlines()
    found = False
    for i, ln in enumerate(lines):
        if "partition_filters:" in ln:
            indent = ln[: len(ln) - len(ln.lstrip())]
            lines[i] = (f'{indent}partition_filters: '
                        f'\'pc.is_in(pc.field("corpus"), pa.array(["{corpus}"]))\'')
            found = True
    if not found:
        raise ValueError("config has no `partition_filters:` line to swap")
    return "\n".join(lines) + "\n"

def parse_metrics(text: str):
    def last_match(patterns):
        val = None
        for pat in patterns:
            m = pat.findall(text)
            if m:
                val = float(m[-1])
        return val
    return last_match(WER_PATTERNS), last_match(UER_PATTERNS)

def run_eval(config_path: Path, out_dir: Path, device: str, log_path: Path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    cmd = ["python", "-m", "workflows.recipes.wav2vec2.asr.eval",
           str(out_dir), "--config-file", str(config_path)]
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout)
    return proc.returncode, proc.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-config",   required=True, type=Path)
    ap.add_argument("--base-config", required=True, type=Path)
    ap.add_argument("--device",   default="0")
    ap.add_argument("--work-dir", type=Path, default=Path("eval_runs"))
    ap.add_argument("--out-csv",  type=Path, default=Path("eval_results.csv"))
    ap.add_argument("--models", nargs="*", default=["finetuned", "base"],
                    help="subset to run; default both")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    cfg_for = {"finetuned": args.ft_config, "base": args.base_config}

    results = []
    for model in args.models:
        cfg_path = cfg_for[model]
        if not cfg_path.exists():
            print(f"!! {model} config not found: {cfg_path}; skipping")
            continue
        base_text = cfg_path.read_text()
        for corpus, label in CORPORA:
            tag = f"{model}_{corpus}"
            print(f"\n========== {tag} ==========")
            cfg_text = set_partition_filter(base_text, corpus)
            tmp_cfg  = args.work_dir / f"cfg_{tag}.yaml"
            tmp_cfg.write_text(cfg_text)
            out_dir  = args.work_dir / f"out_{tag}"
            log_path = args.work_dir / f"log_{tag}.log"

            rc, output = run_eval(tmp_cfg, out_dir, args.device, log_path)
            wer, uer = parse_metrics(output)
            ok = rc == 0 and wer is not None
            status = "ok" if ok else "FAILED"
            wstr = f"{wer:.2f}" if wer is not None else "?"
            print(f"  -> {status}: WER={wstr}  UER={uer}  (exit {rc})")
            if not ok:
                print(f"     see log: {log_path}")
            results.append((model, label, wer, uer, status))

    print("\n" + "=" * 58)
    print(f"{'model':<11}{'test set':<14}{'WER%':>9}{'UER%':>9}{'status':>11}")
    print("-" * 58)
    for m, c, wer, uer, st in results:
        w = f"{wer:.2f}" if wer is not None else "-"
        u = f"{uer:.2f}" if uer is not None else "-"
        print(f"{m:<11}{c:<14}{w:>9}{u:>9}{st:>11}")
    print("=" * 58)

    by = {(m, c): wer for m, c, wer, _, _ in results}
    labels = [lbl for _, lbl in CORPORA]
    have_both = any((("base", l) in by and ("finetuned", l) in by) for l in labels)
    if have_both:
        print("\nbase -> fine-tuned (WER, lower is better):")
        for l in labels:
            b, f = by.get(("base", l)), by.get(("finetuned", l))
            if b is not None and f is not None:
                d = f - b
                arrow = "improved" if d < 0 else "worse"
                print(f"  {l:<14} {b:6.2f} -> {f:6.2f}   ({d:+.2f}, {arrow})")

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "test_set", "wer_pct", "uer_pct", "status"])
        w.writerows(results)
    print(f"\nsaved -> {args.out_csv}")

if __name__ == "__main__":
    main()