from __future__ import annotations

import json
from pathlib import Path

METADATA_PATTERNS = (
    "：",
    " : ",
    " - ",
)
COPYRIGHT_PATTERNS = (
    "QQ音乐享有",
    "TME享有",
    "文曲大模型",
)


def _ms_to_tuple(ms: int) -> tuple[int, float]:
    m = int(ms // 60000)
    s = (ms - m * 60000) / 1000.0
    return m, float(s)


def parse_qrc(qrc_text: str, trans_text: str | None = None) -> list:
    lines = []
    for raw in qrc_text.splitlines():
        li = raw.find("[")
        ri = raw.find("]")
        if li == -1 or ri == -1 or ri <= li + 1:
            continue
        time_part = raw[li + 1 : ri]
        tps = time_part.split(",")
        if len(tps) != 2:
            continue
        try:
            start_ms = int(tps[0])
            length_ms = int(tps[1])
        except Exception:
            continue
        content_part = raw[ri + 1 :]
        words = []
        if content_part:
            for chunk in content_part.split(")"):
                if not chunk:
                    continue
                sp = chunk.split("(")
                if len(sp) != 2:
                    continue
                text = sp[0]
                tt = sp[1].split(",")
                if len(tt) != 2:
                    continue
                try:
                    w_start_ms = int(tt[0])
                    w_len_ms = int(tt[1])
                except Exception:
                    continue
                words.append({"start_ms": w_start_ms, "length_ms": w_len_ms, "content": text})
        line_content = "".join(w["content"] for w in words)
        if not line_content.strip():
            continue
        if any(p in line_content for p in METADATA_PATTERNS):
            continue
        if any(p in line_content for p in COPYRIGHT_PATTERNS):
            continue
        lines.append({"start_ms": start_ms, "length_ms": length_ms, "words": words, "content": line_content})

    labels = []
    for ln in lines:
        if not ln.get("words"):
            continue
        s_ms = ln["start_ms"]
        e_ms = s_ms + ln["length_ms"]
        labels.append({"start": _ms_to_tuple(s_ms), "end": _ms_to_tuple(e_ms), "content": ln.get("content", "")})
    return labels


def trans_qrc_to_labels(src_path: Path) -> dict:
    total = 0
    written = 0
    skipped = 0
    (src_path / "perfect").mkdir(parents=True, exist_ok=True)
    for p in Path(src_path).rglob("*.qrc"):
        if not p.is_file():
            continue
        total += 1
        try:
            qrc_text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            skipped += 1
            print(f"Error reading {p}")
            continue
        labels = parse_qrc(qrc_text)
        if not labels:
            skipped += 1
            print(f"Empty labels for {p}")
            continue
        has_gap = False
        prev_end_ms: int | None = None
        for it in labels:
            s_ms = int(round(((int(it["start"][0]) * 60.0) + float(it["start"][1])) * 1000.0))
            e_ms = int(round(((int(it["end"][0]) * 60.0) + float(it["end"][1])) * 1000.0))
            if prev_end_ms is not None and s_ms > prev_end_ms:
                has_gap = True
                break
            prev_end_ms = e_ms
        out_dir = src_path / ("perfect" if has_gap else "")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (p.stem + ".json")
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
    parser.add_argument("--src", type=str, default=str(Path(".") / "labels_qrc"))
    args = parser.parse_args()
    stats = trans_qrc_to_labels(Path(args.src))
    print(json.dumps({"labels": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
