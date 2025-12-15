import json
import math

import DALI as dali_code

dali_data_path = 'D:\\meliris\\dali\\DALI_v1.0'
dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])

# print(dali_data['e186227bb7474fa5a7738c9108f11972'].info['scores']['NCC'])
# 0.8428142553935238

map_with_NCC = {}
for key, value in dali_data.items():
    ncc = value.info['scores']['NCC']
    if (ncc is None):
        ncc = 0
    # 0.9 以下取小数点后两位，0.9 以上取小数点后一位
    ncc_class = math.trunc(ncc * 100) if ncc < 0.9 else (math.trunc(ncc * 10) * 10)

    if (ncc_class not in map_with_NCC):
        map_with_NCC[ncc_class] = []
    map_with_NCC[ncc_class].append(key)

json.dump(map_with_NCC, 
        open('./dali/map_with_NCC.json', 'w'), 
        indent=4, sort_keys=True)
