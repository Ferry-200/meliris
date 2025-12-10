from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parent.parent / "dali" / "DALI_v1.0_audios"
    parser.add_argument("--root", type=str, default=str(default_root))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root)
    pat = re.compile(r"^(.*)\.([^.]+)\.\2$", re.IGNORECASE)
    planned = []
    count = 0

    for p in root.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        dst = p.with_name(f"{m.group(1)}.{m.group(2)}")
        planned.append((p.name, dst.name))
        if not args.dry_run:
            if dst.exists():
                continue
            p.rename(dst)
        count += 1

    if args.dry_run:
        print(json.dumps({"total": count, "preview": planned[: args.preview]}, ensure_ascii=False))
    else:
        print(count)


if __name__ == "__main__":
    main()

