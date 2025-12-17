import json
import math

import DALI as dali_code

dali_data_path = 'D:\\meliris\\dali\\DALI_v1.0'
dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])

# class Annotations(object):
#     """Basic class that store annotations and its information.

#     It contains some method for transformin the annot representation.
#     """

#     def __init__(self, i=u'None'):
#         self.info = {'id': i, 'artist': u'None', 'title': u'None',
#                      'audio': {'url': u'None', 'working': False,
#                                'path': u'None'},
#                      'metadata': {}, 'scores': {'NCC': 0.0, 'manual': 0.0},
#                      'dataset_version': 0.0, 'ground-truth': False}
#         self.annotations = {'type': u'None', 'annot': {},
#                             'annot_param': {'fr': 0.0, 'offset': 0.0}}
#         self.errors = None
#         return

# print(dali_data['e186227bb7474fa5a7738c9108f11972'].info["audio"])

map_with_NCC = {}
for key, value in dali_data.items():
    ncc = value.info['scores']['NCC']
    if (ncc is None):
        ncc = 0
    # 0.9 以下取小数点后两位，0.9 以上取小数点后一位
    ncc_class = math.trunc(ncc * 100) if ncc < 0.9 else (math.trunc(ncc * 10) * 10)

    if (ncc_class not in map_with_NCC):
        map_with_NCC[ncc_class] = []
    map_with_NCC[ncc_class].append([key, value.info["audio"]["url"]])

json.dump(map_with_NCC, 
        open('./dali/map_with_NCC.json', 'w'), 
        indent=4)
