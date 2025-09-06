
import argparse
from gis.tile import download_tile
from gis.config import Config
import os 
import numpy as np 

config = Config()

def load_mask(coord):
    z,x,y = coord
    filepath = config.mnt_path / f'label/18/{x}_{y}.npy'
    mask = np.load(filepath)
    return mask


def expand_around_tracks():
        
    labels = [x for x in os.listdir(config.mnt_path / 'label/18')]
    labels

    coords = []
    for label in labels:
        x,y = label.split('_')
        x, y = int(x), int(y.replace('.npy', ''))
        coords.append((18, x, y))

    to_expand = [] 
    for coord in coords:
        mask = load_mask(coord)
        has_track = np.mean(mask) > 0
        if has_track:
            to_expand.append(coord)

    EXPANSION_FACTOR = 1
    new_coords = []
    for (z,x,y) in to_expand:
        
        for i in range(-EXPANSION_FACTOR, EXPANSION_FACTOR+1, 1):
            for j in range(-EXPANSION_FACTOR, EXPANSION_FACTOR+1, 1):
                new_coords.append((z, x+i, y+j))
    new_coords = list(set(new_coords))
    print(f'num new coords = {len(new_coords)}')

    new_count = 0
    for coord in new_coords:
        z, x, y = coord
        saved = download_tile(z, x, y)
        if saved:
            new_count += 1
    print(f'added {new_count} new images')




def push_new_imagery():
    print('This is handled in ui of label studio for now ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', 
                       choices=['expand_imagery', 'push_imagery', 'evaluate'], 
                       required=True,
                       help='Mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'expand_imagery':
        expand_around_tracks()
    elif args.mode == 'push_imagery':
        push_new_imagery()