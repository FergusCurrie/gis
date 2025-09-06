

'''
WIth a created export, downlaod annotations and convert
annotations = get_annotations(export_pk=1)
tile2rle = get_tile2rle(annotations)
tile2mask = get_tile2mask(tile2rle)
save_tiles(tile2mask)
tile2mask
'''


import numpy as np
from gis.config import Config
import requests 
import json 

config = Config()



class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i+size]
        self.i += size
        return int(out, 2)
    

def access_bit(data, num):
    """ from bytes array to bits by num position
    """
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift

def bytes2bit(data):
    """ get bit string from bytes data
    """
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def decode_rle(rle):
    """ from LS RLE to numpy uint8 3d image [width, height, channel]
    """
    input = InputStream(bytes2bit(rle))
    num = input.read(32)
    word_size = input.read(5) + 1
    rle_sizes = [input.read(4)+1 for _ in range(4)]
    # print('RLE params:', num, 'values', word_size, 'word_size', rle_sizes, 'rle_sizes')
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input.read(1)
        j = i + 1 + input.read(rle_sizes[input.read(2)])
        if x:
            out[i:j] = input.read(word_size)
            i = j
        else:
            while i < j:
                val = input.read(word_size)
                out[i] = val
                i += 1
    return out.reshape((256,256, 4))[..., 1]


from label_studio_sdk import LabelStudio

def get_annotations(export_pk: int):
    client = LabelStudio(
        base_url=config.label_studio_url,
        api_key=config.label_studio_token,
    )
    return [a for a in client.projects.exports.download(export_pk=export_pk, id=4)]

def get_tile2rle(annotations: list):
    tile2rle = {}
    join_annotations = b"".join(annotations)
    annotations_decoded = json.loads(join_annotations.decode("utf-8"))
    for annotation in annotations_decoded:
        input_data_file: str = annotation['data']['image']
        z, x, y = input_data_file.split('/')[-1].split('_')
        z, x, y = int(z), int(x), int(y.replace('.jpg', ''))

        ann: list = annotation['annotations']
        if ann: 
            results: list = ann[0]['result']
            if results:
                tile2rle[(z,x,y)] = {'rle':results[0]['value']['rle'], 'empty': False}
            else:
                tile2rle[(z,x,y)] = {'rle' : [], 'empty': True}
    return tile2rle

def get_tile2mask(tile2rle: dict):
    tile2mask = {}
    for (z,x,y), rle_dict in tile2rle.items():
        if rle_dict['empty']:
            tile2mask[(z,x,y)] = np.zeros((256,256), dtype=np.uint8)
        else:
            tile2mask[(z,x,y)] = decode_rle(bytes(rle_dict['rle']))
    return tile2mask


def save_tiles(tile2mask: dict) -> None:
    for (z,x,y), mask in tile2mask.items():
        path = config.mnt_path / f'label/{z}/{x}_{y}.npy'
        np.save(file=path, arr=mask)


def create_export_snapshot(project_id, export_type='JSON', filters=None):
    """Create an export snapshot (starts background task)"""
    url = f"{config.label_studio_url}/api/projects/{project_id}/exports/"
    
    payload = {'export_type': export_type}
    if filters:
        payload.update(filters)

    headers = {
        'Authorization': f'Token {config.label_studio_token}',
        'Content-Type': 'application/json'
    }
    
    print(f"Creating export snapshot for project {project_id}...")
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    export_info = response.json()
    print(f"Export snapshot created with ID: {export_info['id']}")
    return export_info
