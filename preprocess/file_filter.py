# %%
import os
from os import path as osp
import re
import shutil
from PIL import Image
from glob import glob
from typing import List
from tqdm.auto import tqdm
from itertools import chain
import argparse
import settings
import json
from datasets import load_dataset
import logging
from utils import load_config_from_file, save_setting_to_file

logger = logging.getLogger(__name__)

image_format = ["jpeg", "jpg", "png", "webp"]

# %%
search_exclude_pairs = [
    (["mecha", "robot", "mecha musume"], ["girl", "boy"]),
    (None, ["sensitive", "explicit"]),
]

hashList = []
count = 0


# %%
def imgFilter(words):
    """
    Filter the given list of words based on search and exclude conditions.

    Args:
        words (list): List of words to be filtered.

    Returns:
        bool: True if the content meets the search condition and not the exclude condition, False otherwise.
    """
    # Initialize a flag to track whether the content should be included
    include = False

    # Iterate over the search/exclude pairs
    for search_words, exclude_words in search_exclude_pairs:
        # create a regex pattern to match the word as a whole word
        has_search_words = None
        has_exclude_words = None

        if search_words:
            pattern = "|".join([r"{}\b".format(word) for word in search_words])
            # Check if any of the search words are in the list
            has_search_words = any(re.search(pattern, word) for word in words)

        if exclude_words:
            pattern = "|".join([r"{}\b".format(word) for word in exclude_words])
            # Check if any of the exclude words are in the list, or if the exclude list is empty
            has_exclude_words = any(re.search(pattern, word) for word in words)

        # If the content meets the search condition, set include to True
        if has_search_words and not has_exclude_words:
            include = True
        elif has_search_words and has_exclude_words:
            include = False
        elif has_search_words is None and has_exclude_words:
            include = False
        # does not check has_search_words is none and has_exclude_words
        # this is to make (None, ["sensitive", "explicit"]) work

    # If none of the search conditions are met, return False
    return include


def add_custom_tag(tag_list: List[str], custom_tags: str) -> List[str]:
    """
    Add custom tags to the tag list.

    Args:
        tag_list (List[str]): The original list of tags.
        custom_tags (str): The custom tags to be added.

    Returns:
        List[str]: The updated list of tags.
    """
    # Split the custom tags string by comma and strip any leading or trailing whitespace
    new_tags = [t.strip() for t in custom_tags.split(",") if t.strip() != ""]

    # Combine the original tag list with the new tags
    updated_tag_list = tag_list + new_tags

    return updated_tag_list


def filter_images_by_resolution(image_path: str, min_resolution: int):
    """
    Function to filter images based on their resolution.

    Args:
        image_path (str): The path of the image file.
        min_resolution (int): The minimum resolution required for the image.

    Returns:
        bool: True if the image resolution is smaller than the minimum resolution, False otherwise.
    """
    with Image.open(image_path) as img:
        resolution = img.size
        if resolution[0] < min_resolution or resolution[1] < min_resolution:
            return True
    return False


# %%
def main(
    src_path: str,
    dst_path: str,
    tag_extension: str,
    caption_extension: str,
    debug_dir: str = None,
    config: str = None,
):
    # Load config if provided
    if config:
        load_config_from_file(config)

    # Remove existing destination directory if it exists
    if osp.isdir(dst_path):
        shutil.rmtree(dst_path)

    # Create destination directory
    os.makedirs(dst_path, exist_ok=True)
    
    # Saving current batch setting to file
    save_setting_to_file(osp.join(dst_path, "settings.json"))

    # Calculate aesthetic scores if filtering using cafe aesthetic
    if settings.filter_using_cafe_aesthetic:
        from cafe_filter import Aesthetic

        scorer = Aesthetic(batch_size=settings.cafe_batch_size, aesthetic=True)

    # Find all image files in the source path
    imgList = list(
        chain(*[glob(osp.join(src_path, f"*.{f}")) for f in image_format])
    )
    logger.info(f"find {len(imgList)} image file")

    imgList = [
        image
        for image in imgList
        if not filter_images_by_resolution(image, settings.filter_min_resolution)
    ]
    logger.info(f"{len(imgList)} image file met min resolution")
    

    # Initialize output lists
    output = []
    debug_output = []

    # Calculate aesthetic scores if filtering using cafe aesthetic
    if settings.filter_using_cafe_aesthetic:
        logger.info("Calculating aesthetics...")
        dataset = load_dataset("imagefolder", data_files=imgList)
        scores = scorer.calculate_aesthetic_score(dataset["train"])
        logger.info(f"Finish calculating aesthetic")

    # Filter and process images
    for idx, imgFile in tqdm(enumerate(imgList), desc="filter"):
        try:
            id = osp.splitext(osp.basename(imgFile))[0]
            tagFile = osp.join(src_path, f"{id}{tag_extension}")
            tagOri = osp.join(src_path, f"{id}.txt")
            caption_src = osp.join(src_path, f"{id}{caption_extension}")

            # Calculate score based on cafe aesthetics
            score = (
                int(scores[idx]["aesthetic"] * 100)
                if settings.filter_using_cafe_aesthetic
                else 100
            )

            with open(tagFile, "r") as f:
                tags = f.read()
            tags = [t.strip() for t in tags.split(",")]

            # Filter images based on tags and score
            if imgFilter(tags) and score >= 70:
                output.append(
                    {
                        "img_src": imgFile,
                        "tag_ori": tagOri,
                        "tag_src": tagFile,
                        "tags": tags,
                        "caption_src": caption_src,
                        "id": id,
                    }
                )
            elif debug_dir:
                debug_output.append(
                    {
                        "img_src": imgFile,
                        "id": id,
                        "reason": "tag/score",
                        "tags": tags,
                        "score": score,
                    }
                )
        except:
            logger.info(f"Failed to process image {imgFile}")
            debug_output.append({"img_src": imgFile, "id": id, "reason": "error"})

    # Process filtered images
    for idx, item in enumerate(output):
        tag_list = item["tags"]
        if hasattr(settings, "custom_tags") and settings.custom_tags is not None:
            tag_list = add_custom_tag(tag_list, settings.custom_tags)
        tag_list = list(set(tag_list))
        with open(item["tag_src"], "w") as f:
            f.write(",".join(tag_list))

        imgFile = item["img_src"]
        img_dst = osp.join(dst_path, osp.basename(imgFile))
        os.symlink(imgFile, img_dst)

        tag_dst = osp.join(dst_path, f"{item['id']}{tag_extension}")
        if settings.use_original_tags:
            tag_ori_dst = osp.join(dst_path, f"{item['id']}.txt")
            os.symlink(item["tag_ori"], tag_ori_dst)
        else:
            os.symlink(item["tag_src"], tag_dst)

        if osp.isfile(item["caption_src"]):
            caption_dst = osp.join(dst_path, f"{item['id']}{caption_extension}")
            os.symlink(item["caption_src"], caption_dst)

    if debug_dir:
        os.makedirs(osp.join(debug_dir, "filter"), exist_ok=True)
        for item in debug_output:
            imgFile = item["img_src"]
            img_dst = osp.join(debug_dir, "filter", osp.basename(imgFile))
            os.symlink(imgFile, img_dst)
            with open(osp.join(debug_dir, "filter", f"{item['id']}.json"), "w") as f:
                json.dump(item, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="src dir")
    parser.add_argument("--dst", type=str, help="dst dir")
    parser.add_argument("--tag_extension", type=str, default=".tag", help="dst dir")
    args = parser.parse_args()
    main(args.src, args.dst, args.tag_extension)


# %%
