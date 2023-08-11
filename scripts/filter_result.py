#%%
import os
import os.path as osp
from glob import glob
import json
from tqdm import tqdm
import click

@click.command(help='Filter result')
@click.option('--image_path', type=str, help='Path to the Images', prompt='Enter the image path:', show_default=True, default='')
@click.option('--exclude_json', type=str, help='Path to exclude.json', prompt='Enter the path to exclude.json:', show_default=True, default='')

def main(image_path, exclude_json):
    if osp.isfile(exclude_json):
        with open(exclude_json, "r") as f:
            exclude = json.load(f)
        files_in_dir = glob(osp.join(image_path, "*"))
        for item in tqdm(exclude):
            filename = osp.splitext(item)[0] + "."
            files_to_delete = [f for f in files_in_dir if filename in f]
            for f in files_to_delete:
                if osp.isfile(f):
                    os.remove(f)
    else:
        print("exclude.json not found")
    
    meta_file = osp.join(image_path, "meta_lat.json")
    if osp.isfile(meta_file):
        meta_lat = json.load(open(meta_file,'r'))
        for item in exclude:
            file = os.path.splitext(item)[0]
            del meta_lat[file]
        with open(meta_file, 'w') as f:
            json.dump(meta_lat, f, indent=4)

if __name__ == "__main__":
    main()
