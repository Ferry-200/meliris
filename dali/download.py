import DALI as dali_code
import json
import re
from pathlib import Path
dali_data_path = 'D:\\meliris\\dali\\DALI_v1.0'
dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])

dali_info = dali_code.get_info(dali_data_path + '\\info\\DALI_DATA_INFO.gz')
"""
[['DALI_ID' 'NAME' 'YOUTUBE' 'WORKING']
 ['e186227bb7474fa5a7738c9108f11972' 'Staind-Tangled_Up_In_You'
  'NXG-ayocugI' 'True']
 ['520f583def024997adcab0567fb25a5d' 'Boyzone-Baby_Can_I_Hold_You'
  'ZjSLNZ9MsMI' 'True']
 ['0f46aeae45ed4e6987f8b35e40d96c59' 'The_Killers-For_Reasons_Unknown'
  'TG5X4kOjEX8' 'True']]
"""
dali_info = dali_info[0:802]

dali_audios_output = "D:\\meliris\\dali\\DALI_v1.0_audios"

out_root = Path(dali_audios_output)
out_root.mkdir(parents=True, exist_ok=True)

def _collect_downloaded_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for pp in root.rglob("*"):
        if pp.is_file():
            m = re.match(r"^[0-9a-f]{32}", pp.name)
            if m:
                ids.add(m.group(0))
    return ids

skip_ids = _collect_downloaded_ids(out_root)
dali_code.get_audio(dali_info, dali_audios_output, cookiefile="D:\\meliris\\dali\\www.youtube.com_cookies.txt", skip=list(skip_ids))
