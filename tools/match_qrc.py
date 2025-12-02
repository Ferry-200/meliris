import asyncio
import json
import random
import re
from pathlib import Path
from difflib import SequenceMatcher

from qqmusic_api import search, lyric


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _sim(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _read_meta(path: Path) -> dict:
    title = ""
    artist = ""
    album = ""
    try:
        from mutagen import File as MFile
        from mutagen.id3 import ID3
    except Exception:
        return {"title": path.stem, "artist": "", "album": ""}
    audio = MFile(str(path))
    if audio is None:
        return {"title": path.stem, "artist": "", "album": ""}
    tags = getattr(audio, "tags", None)
    try:
        if isinstance(tags, ID3):
            f = tags.get("TIT2")
            if f and hasattr(f, "text") and f.text:
                title = str(f.text[0])
            f = tags.get("TPE1")
            if f and hasattr(f, "text") and f.text:
                artist = "/".join([str(x) for x in f.text]) if isinstance(f.text, list) else str(f.text)
            f = tags.get("TALB")
            if f and hasattr(f, "text") and f.text:
                album = str(f.text[0])
        else:
            if tags:
                for k in ("TITLE", "title"):
                    if k in tags and tags[k]:
                        v = tags[k]
                        title = v[0] if isinstance(v, list) else str(v)
                        break
                for k in ("ARTIST", "artist"):
                    if k in tags and tags[k]:
                        v = tags[k]
                        artist = v[0] if isinstance(v, list) else str(v)
                        break
                for k in ("ALBUM", "album"):
                    if k in tags and tags[k]:
                        v = tags[k]
                        album = v[0] if isinstance(v, list) else str(v)
                        break
    except Exception:
        pass
    if not title:
        title = path.stem
    return {"title": title, "artist": artist, "album": album}


async def _search_best(keyword: str, artist: str, album: str, limit: int = 10) -> dict | None:
    res = await search.search_by_type(keyword=keyword, num=limit)
    best = None
    best_score = -1.0
    for item in res or []:
        name = _norm(item.get("name"))
        singers = []
        try:
            singers = [x.get("name", "") for x in item.get("singer", [])]
        except Exception:
            singers = []
        # singer_str = _norm("/".join(singers))
        album_name = _norm((item.get("album") or {}).get("name"))
        s = 0.6 * _sim(_norm(keyword), name) + 0.3 * max(_sim(_norm(artist), _norm(x)) for x in singers) if singers else 0.0 + 0.1 * _sim(_norm(album), album_name)
        if s > best_score:
            best_score = s
            best = item
    return best


async def process_dir(src: Path, out_root: Path, limit: int = 5) -> dict:
    def _collect_files(root: Path) -> list[Path]:
        files = []
        for pp in root.rglob("*"):
            if pp.is_file() and pp.suffix.lower() in {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".aac", ".wma"}:
                files.append(pp)
        files.sort(key=lambda x: str(x).lower())
        return files

    total = 0
    written = 0
    skipped = 0
    out_root.mkdir(parents=True, exist_ok=True)
    resume_path = out_root / ".resume.json"
    files = _collect_files(src)
    total = len(files)

    start_idx = 0
    if resume_path.exists():
        try:
            state = json.loads(resume_path.read_text(encoding="utf-8"))
            cur = state.get("current")
            if cur:
                for i, fp in enumerate(files):
                    if str(fp) == str(cur):
                        start_idx = i
                        break
        except Exception:
            start_idx = 0

    try:
        for i in range(start_idx, len(files)):
            p = files[i]
            resume_path.write_text(json.dumps({"current": str(p), "index": i, "total": total}, ensure_ascii=False), encoding="utf-8")
            print(f"Processing {p}")

            meta = _read_meta(p)
            kw = meta.get("title")
            if meta.get("artist"):
                kw = f"{kw} {meta.get('artist')}"
            print(f"Searching for {kw}")

            await asyncio.sleep(random.uniform(1, 3))
            candidate = await _search_best(kw, meta.get("artist", ""), meta.get("album", ""), limit=limit)

            if not candidate:
                skipped += 1
                print(f"Empty candidate for {kw} {p}")
                continue
            mid = candidate.get("mid") or (candidate.get("file") or {}).get("media_mid")
            print(f"Found {mid}")

            if not mid:
                skipped += 1
                print(f"Empty mid for {kw} {p}")
                continue

            await asyncio.sleep(random.uniform(1, 3))
            lr = await lyric.get_lyric(mid, qrc=True)
            qrc_text = (lr or {}).get("lyric") or ""
            if not qrc_text.strip():
                skipped += 1
                print(f"Empty lyric for {mid} {p}")
                continue
            out_path = out_root / (p.stem + ".qrc")
            out_path.write_text(qrc_text, encoding="utf-8")
            written += 1
            print(f"Processed {i + 1} / {total}, written {written}, skipped {skipped}")
    except KeyboardInterrupt:
        print("Interrupted, resume state saved.")
    except Exception as e:
        print(f"Error: {e}. Resume state saved.")
    finally:
        try:
            if written + skipped >= total:
                if resume_path.exists():
                    resume_path.unlink()
        except Exception:
            pass
    return {"total": total, "written": written, "skipped": skipped}


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(Path(r"D:\meliris\music")))
    parser.add_argument("--out", type=str, default=str(Path(".") / "labels_qrc" / "raw"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out)
    if args.start:
        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / ".resume.json").write_text(json.dumps({"current": args.start}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    elif not args.resume:
        try:
            if (out_root / ".resume.json").exists():
                (out_root / ".resume.json").unlink()
        except Exception:
            pass
    stats = await process_dir(Path(args.src), out_root, limit=int(args.limit))
    print(json.dumps({"labels_qrc": stats}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
