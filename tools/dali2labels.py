from __future__ import annotations

import json
from pathlib import Path


def _sec_to_tuple(sec: float) -> tuple[int, float]:
    m = int(sec // 60)
    s = sec - m * 60
    return m, float(s)


def parse_dali_lines(entry: dict) -> list:
    annot = entry.get("annotations", {}).get("annot", {})
    lines = annot.get("lines") or []
    labels = []
    for ln in lines:
        t = ln.get("time")
        if not (isinstance(t, (list, tuple)) and len(t) == 2):
            continue
        try:
            s_sec = float(t[0])
            e_sec = float(t[1])
        except Exception:
            continue
        text = str(ln.get("text", ""))
        if not text.strip():
            continue
        labels.append({
            "start": _sec_to_tuple(s_sec),
            "end": _sec_to_tuple(e_sec),
            "content": text,
        })
    return labels


def trans_dali_to_labels(src_path: Path, out_root: Path | None = None) -> dict:
    total = 0
    written = 0
    skipped = 0
    out_root = out_root or (Path(".") / "labels_dali")
    out_root.mkdir(parents=True, exist_ok=True)
    for p in Path(src_path).rglob("*.json"):
        if not p.is_file():
            continue
        total += 1
        try:
            data_text = p.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(data_text)
        except Exception:
            skipped += 1
            print(f"Error reading {p}")
            continue
        labels = []
        if isinstance(data, dict):
            if "annotations" in data:
                labels = parse_dali_lines(data)
            else:
                for v in data.values():
                    if isinstance(v, dict) and "annotations" in v:
                        labels = parse_dali_lines(v)
                        break
        elif isinstance(data, list):
            for v in data:
                if isinstance(v, dict) and "annotations" in v:
                    labels = parse_dali_lines(v)
                    break
        if not labels:
            skipped += 1
            print(f"Empty labels for {p}")
            continue
        out_path = out_root / (p.stem + ".json")
        try:
            out_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
            written += 1
        except Exception:
            skipped += 1
            print(f"Error writing {out_path}")
            continue
    return {"total": total, "written": written, "skipped": skipped}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    default_src = Path(__file__).resolve().parent.parent / "dali" / "DALI_v1.0_audios"
    parser.add_argument("--src", type=str, default=str(default_src))
    parser.add_argument("--out", type=str, default=str(Path(".") / "labels_dali"))
    args = parser.parse_args()
    stats = trans_dali_to_labels(Path(args.src), Path(args.out))
    print(json.dumps({"labels": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()

