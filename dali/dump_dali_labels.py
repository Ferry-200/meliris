import DALI as dali_code
import json
import re
from pathlib import Path

def _collect_downloaded_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for pp in root.rglob("*"):
        if pp.is_file():
            m = re.match(r"^[0-9a-f]{32}", pp.name)
            if m:
                ids.add(m.group(0))
    return ids

dali_audios_output = "D:\\meliris\\dali\\DALI_v1.0_audios"
out_root = Path(dali_audios_output)
downloaded_ids = _collect_downloaded_ids(out_root)

print(f"Total {len(downloaded_ids)} downloaded")

dali_data_path = 'D:\\meliris\\dali\\DALI_v1.0'
dali_data = dali_code.get_the_DALI_dataset(dali_data_path)

print(f"Total {len(dali_data)} entries")

for dali_id in downloaded_ids:
    entry = dali_data[dali_id]
    print(f"Exporting {dali_id}")
    entry.write_json(dali_audios_output, dali_id)
