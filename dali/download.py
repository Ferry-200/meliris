import DALI as dali_code
import json
import re
from pathlib import Path

dali_data_path = 'D:\\meliris\\dali\\DALI_v1.0'
dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])

ncc_map = json.load(open('./dali/map_with_NCC.json', 'r'))
# [["DALI_ID", "YOUTUBE_ID"]]
great_dali_items = []
# collect ncc 90, 89, 88 items
# for item in ncc_map["90"]:
#     great_dali_items.append(item)
# for item in ncc_map["89"]:
#     great_dali_items.append(item)
# for item in ncc_map["88"]:
#     great_dali_items.append(item)
for item in ncc_map["87"]:
    great_dali_items.append(item)

print(f"try to download {len(great_dali_items)} items.")

dali_audios_output = "D:\\meliris\\dali\\DALI_v1.0_audios"

out_root = Path(dali_audios_output)
out_root.mkdir(parents=True, exist_ok=True)

def collect_downloaded_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for pp in root.rglob("*"):
        if pp.is_file():
            m = re.match(r"^[0-9a-f]{32}", pp.name)
            if m:
                ids.add(m.group(0))
    return ids

skip_ids = collect_downloaded_ids(out_root)
for great_item in great_dali_items:
    if (great_item[0] in skip_ids):
        print(f"----- skip downloaded item {great_item[0]} -----")
        continue
    print(f"----- try to download {great_item[0]} -----")
    dali_code.audio_from_url(great_item[1], great_item[0], dali_audios_output, 
                    cookiefile="D:\\meliris\\dali\\www.youtube.com_cookies.txt")
