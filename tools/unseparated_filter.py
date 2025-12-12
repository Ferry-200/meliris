import re
from pathlib import Path

def _collect_all_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for pp in root.rglob("*"):
        if pp.is_file():
            m = re.match(r"^[0-9a-f]{32}", pp.name)
            if m:
                ids.add(m.group(0))
    return ids

dali_audios_output = r"D:\meliris\dali\DALI_v1.0_audios"
out_root = Path(dali_audios_output)

all = _collect_all_ids(out_root)

print(len(all))
pending_folder = out_root / "pending"
pending_folder.mkdir(parents=True, exist_ok=True)

separated_root = Path(r"D:\ferry\Demucs-GUI_1.3.2_cuda_mkl\separated\htdemucs")

for id in all:
    separated_folder = separated_root / id
    if not separated_folder.exists():
        print(f"copy {id}.mp3 to pending folder")
        (pending_folder / (id + ".mp3")).touch()
        (pending_folder / (id + ".mp3")).write_bytes((out_root / (id + ".mp3")).read_bytes())
