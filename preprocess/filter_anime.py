#%%
import sys
import os
from os import path as osp
from glob import glob, escape
from utils import *
import shutil
from pathlib import Path
path_to_repo = "/notebooks/kohya-trainer-paperspace"
if osp.join(path_to_repo,'finetune') not in sys.path:
    sys.path.append(osp.join(path_to_repo,'finetune'))
import tag_images_by_wd14_tagger

data_dir = '/data'
table = str.maketrans({"[":  r"\[", "]":  r"\]", " ": r"\ ", "^":  r"\^", "$":  r"\$", "*":  r"\*", "&": r"\&"})

keywords = ['mecha', 'robot', 'mecha_musume']
def filter_image_by_tag(folder, dst_folder):
    if osp.isfile(osp.join(dst_folder, 'complete')):
        return
    os.makedirs(dst_folder, exist_ok=True)
    tags = glob(osp.join(folder, '*.tag'))
    for tag_file in tqdm(tags):
        with open(tag_file, 'r') as f:
            tag = f.read().strip()
        tag = [t.strip() for t in tag.split(',')] 
        exist = False
        for keyword in keywords:
            if keyword in tag:
                exist = True
                break
        if exist:
            shutil.move(tag_file.replace(".tag", ".png"), dst_folder)
    Path(osp.join(dst_folder, 'complete')).touch()
            
if __name__ == '__main__':
    dst_root_dir = osp.join(data_dir,'filter')
    image_folders = glob(osp.join(data_dir,"*/*"))
    for folder in image_folders:
        dst_dir = folder.replace(data_dir, dst_root_dir)
        if 'filter' in folder or osp.isfile(osp.join(dst_dir, 'complete')):
            continue
        if not osp.isfile(osp.join(folder, 'complete')):
            print(f"{folder} is not complete, skipping...")
        if len(glob(osp.join(folder, '*.tag'))) != glob(osp.join(folder, '*.png')):      
            print(f"Start processing {folder}")
            wd_args = prepare_wd_parser(folder, thresh=0.4, batch_size=4, caption_extention='.tag')
            tag_images_by_wd14_tagger.main(wd_args)
        print(f"Start filtering {folder}")
        filter_image_by_tag(folder, dst_dir)
        