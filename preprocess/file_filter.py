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
import logging
logger = logging.getLogger(__name__)

image_format = ['jpeg','jpg', 'png', 'webp']

#%%
search_exclude_pairs = [
    (['mecha', 'robot'], ['girl', 'boy']),
    (None, ['sensitive', 'explicit'])
]
hashList = []
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

def add_custom_tag(tag_file, custom_tags):
    with open(tag_file,'r') as f:
        tags = f.read()
    words = [t.strip() for t in tags.split(',')]
    words += [t.strip() for t in custom_tags.split(',') if t.strip() != ""]
    with open(tag_file,'w') as f:
        f.write(','.join(words))
#%%
def main(src_path, dst_path, tag_extension, caption_extension, filter_using_cafe_aesthetic=False, debug_dir=None):

    if osp.isdir(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path,exist_ok=True)
    
    if filter_using_cafe_aesthetic:
        from cafe_filter import Aesthetic
        scorer = Aesthetic(batch_size=settings.cafe_batch_size)
    
    imgList = list(chain(*[glob(os.path.join(src_path, f"*.{f}")) for f in image_format]))
    logger.info(f"find {len(imgList)} image file")
    
    # simple filter
    output = []
    debug_output = []
    for idx, imgFile in tqdm(enumerate(imgList), desc="filter"):
        try:
            id = osp.splitext(osp.basename(imgFile))[0]
            tagFile = osp.join(src_path,f"{id}{tag_extension}")
            tagOri = osp.join(src_path,f"{id}.txt")
            caption_src = osp.join(src_path,f"{id}{caption_extension}")
            with open(tagFile,'r') as f:
                tags = f.read()
            if imgFilter(tags):
                output.append({ 'img_src': imgFile, 
                                 'tag_ori': tagOri,
                                'tag_src': tagFile,
                                'caption_src': caption_src,
                                'id':  id,
                            })
            elif debug_dir:
                debug_output.append({
                                'img_src': imgFile,
                                'id': id,
                                'reason': "tag",
                                'tags': tags
                            })
        except:
            logger.info(f"Failed to process image {imgFile}")
            debug_output[id] = f"Reason: Unknow error;"
            
    if filter_using_cafe_aesthetic:
        final_output = []
        logger.info("Calculating aesthetics...")
        dataset = load_dataset("imagefolder", data_files = [i['img_src'] for i in output])
        scores = scorer.calculate_aesthetic_score(dataset['train'])
        logger.info(f"Finish calculating aesthetic")
        for idx, item in enumerate(output):
            score = scores[idx]
            if score['aesthetic'] < settings.filter_aesthetic_thresh \
                or score['anime'] < settings.filter_anime_thresh \
                    or score['not_waifu'] < settings.filter_waifu_thresh:
                        if debug_dir:
                            debug_output.append({
                                            'img_src': item['img_src'],
                                            'id': item['id'],
                                            'reason': "aesthetic",
                                            'score': score
                                        })
                        continue
            final_output.append(item)
    else:
        final_output = output
    
    for idx, item in enumerate(final_output):
        if hasattr(settings, "custom_tags"):
            add_custom_tag(item['tag_src'], settings.custom_tags)
        imgFile = item['img_src']
        img_dst = osp.join(dst_path, osp.basename(imgFile))
        tag_dst = osp.join(dst_path, f"{item['id']}{tag_extension}")
        tag_ori_dst = osp.join(dst_path, f"{item['id']}.txt")
        caption_dst = osp.join(dst_path, f"{item['id']}{caption_extension}")
        os.symlink(imgFile, img_dst)
        os.symlink(item['tag_src'], tag_dst)
        if settings.use_original_tags:
            os.symlink(item['tag_ori'], tag_ori_dst)
        os.symlink(item['caption_src'], caption_dst)
        
    if debug_dir:
        for item in debug_output:
            imgFile = item['img_src']
            img_dst = osp.join(debug_dir, osp.basename(imgFile))
            os.symlink(imgFile, img_dst)
            with open(osp.join(debug_dir, f"{item['id']}.json"), 'w') as f:
                json.dump(item, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="src dir")
    parser.add_argument("--dst", type=str, help="dst dir")
    parser.add_argument("--tag_extension", type=str, default='.tag', help="dst dir")
    args = parser.parse_args()
    main(args.src,args.dst,args.tag_extension)
    
