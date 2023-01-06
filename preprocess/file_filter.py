#%%
import os
from os import path as osp
import shutil
from glob import glob
import re
from tqdm.auto import tqdm
from itertools import chain
import argparse
from PIL import Image
import settings
import json
from datasets import load_dataset
image_format = ['jpeg','jpg', 'png', 'webp']

#%%
search_exclude_pairs = [
    (['mecha', 'robot'], ['girl', 'boy']),
    (None, ['sensitive', 'explicit'])
]
hashList = []
output = []
count = 0

def imgFilter(tag):
    # Initialize a flag to track whether the content should be included
    include = False
    # Split the input string into a list of words
    words = [t.strip() for t in tag.split(',')]
    # Iterate over the search/exclude pairs
    for search_words, exclude_words in search_exclude_pairs:
        # create a regex pattern to match the word as a whole word
        has_search_words = None; has_exclude_words = None
        if search_words:
            pattern = '|'.join([r"{}\b".format(word) for word in search_words])
            # Check if any of the search words are in the list
            has_search_words =  any(re.search(pattern, word) for word in words) if search_words else None
        if exclude_words:
            pattern = '|'.join([r"{}\b".format(word) for word in exclude_words])
            # Check if any of the exclude words are in the list, or if the exclude list is empty
            has_exclude_words = any(re.search(pattern, word) for word in words) if exclude_words else True
        # If the content meets the search condition, return True
        if has_search_words and not has_exclude_words:
            include = True 
        elif has_search_words and has_exclude_words:
            include = True
        elif has_search_words is None and has_exclude_words:
            include = False
    # If none of the search conditions are met, return False
    return include

#%%

def image_generator(imgs):
    for img in imgs:
        yield Image.open(img)

def main(src_path, dst_path, tag_extension, caption_extension, filter_using_cafe_aesthetic=False, debug_dir=None):
    if osp.isdir(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path,exist_ok=True)
    
    if filter_using_cafe_aesthetic:
        from cafe_filter import Aesthetic
        scorer = Aesthetic(batch_size=settings.cafe_batch_size)
    
    imgList = list(chain(*[glob(os.path.join(src_path, f"*.{f}")) for f in image_format]))
    print(f"find {len(imgList)} image file")
    
    if filter_using_cafe_aesthetic:
        # batch_split = settings.cafe_batch_size
        # scores = []
        dataset = load_dataset("imagefolder", data_files =imgList)
        scores = scorer.calculate_aesthetic_score(dataset['train'])
        # for i in tqdm(range(len(imgList)//batch_split + 1), desc="Calculating aesthetic"):
        #     start_id = i * batch_split
        #     end_id = (i+1) * batch_split if (i+1) * batch_split < len(imgList) else None
        #     imgs = [Image.open(i) for i in imgList[start_id:end_id]]
        #     scores += scorer.calculate_aesthetic_score(imgs)
        print(f"Finish calculating aesthetic")
        
    for idx, imgFile in tqdm(enumerate(imgList), desc="filter"):
        try:
            id = osp.splitext(osp.basename(imgFile))[0]
            tagFile = osp.join(src_path,f"{id}{tag_extension}")
            tagOri = osp.join(src_path,f"{id}.txt")
            caption_src = osp.join(src_path,f"{id}{caption_extension}")
            with open(tagFile,'r') as f:
                tags = f.read()
            if imgFilter(tags):
                if filter_using_cafe_aesthetic:
                    score = scores[idx]
                    if debug_dir:
                        debug_file = osp.join(debug_dir, f"{osp.basename(imgFile)}_aesthetic.json")
                        with open(debug_file, 'w') as f:
                            json.dump(score, f, indent=4)
                    if score['aesthetic'] < settings.filter_aesthetic_thresh \
                        or score['anime'] < settings.filter_anime_thresh \
                            or score['not_waifu'] < settings.filter_waifu_thresh:
                                continue
                    
                output.append({ 'img_src': imgFile, 
                                 'tag_ori': tagOri,
                                'tag_src': tagFile,
                                'caption_src': caption_src,
                                'id':  id,
                            })
        except:
            print(f"Failed to process image {imgFile}")
            
    for idx, item in enumerate(output):
        imgFile = item['img_src']
        img_dst = osp.join(dst_path, osp.basename(imgFile))
        tag_dst = osp.join(dst_path, f"{item['id']}{tag_extension}")
        tag_ori_dst = osp.join(dst_path, f"{item['id']}.txt")
        caption_dst = osp.join(dst_path, f"{item['id']}{caption_extension}")
        os.symlink(imgFile, img_dst)
        os.symlink(item['tag_src'], tag_dst)
        os.symlink(item['tag_ori'], tag_ori_dst)
        os.symlink(item['caption_src'], caption_dst)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="src dir")
    parser.add_argument("--dst", type=str, help="dst dir")
    parser.add_argument("--tag_extension", type=str, default='.tag', help="dst dir")
    args = parser.parse_args()
    main(args.src,args.dst,args.tag_extension)
    
