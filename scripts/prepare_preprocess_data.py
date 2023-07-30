#%%
import re
import sys
import os
from os import path as osp
import argparse
import json
from glob import glob
from tqdm import tqdm
from itertools import chain
from minio import Minio

image_format = ["jpeg", "jpg", "png", "webp"]

def get_first_index_to_use(s3, bucket_name:str):
    response = [
        int(o.object_name.replace(".tar.gz", ""))
        for o in s3.list_objects(bucket_name)
        if "result" not in o.object_name
    ]
    response.sort()
    if len(response) == 0:
        return 0
    return response[-1] + 1

# %%
def main(args):
    files_to_skip = []; hashList = []; output = []
    s3 = Minio(
        "192.168.50.210:9000",
        access_key="Ovzmp8bmq50RTsJg",
        secret_key="rFteTYVeiNedCOTM6pG9nFtRpqu6izld",
        secure=False,
    )
    config = {"use_original_tags": args.source == "booru", "wd14_thresh": args.wd14_thresh,
            "custom_tags": args.custom_tags}
    os.makedirs(args.output_path, exist_ok=True)
    # if source == "pinterest":
    #     history_dir = "pinterest_history"
    #     history_files = glob(osp.join(history_dir, "*.json"))
    #     if len(history_files):
    #         for hf in history_files:
    #             with open(hf, "r") as f:
    #                 files_to_skip += json.load(f)
    
    imgList = list(
        chain(
            *[
                glob(os.path.join(args.image_path, f"*.{f}"))
                for f in image_format
            ]
        )
    )
    print(f"find {len(imgList)} image file")
    
    assert len(imgList) > 0, "No image file found"

    zip_start_index = get_first_index_to_use(s3, args.bucket_name)
    print(f"Will start with index {zip_start_index}")

    if args.source == "booru":
        for imgFile in tqdm(imgList):
            id = osp.splitext(osp.basename(imgFile))[0]
            tagSearch = glob(
                osp.join("/".join(imgFile.split("/")[:-2]), f"tag/{id}.json")
            )
            if len(tagSearch) == 1:
                with open(tagSearch[0], "r") as f:
                    tag = json.load(f)
                if (
                    tag["md5"] in hashList
                    or tag["rating"] == "s"
                    or tag["rating"] == "e"
                    or tag["rating"] == "q"
                ):
                    continue
                hashList.append(tag["md5"])
                tagValues = tag["tag_string_general"].split()
                tagValues += tag["tag_string_copyright"].split()
                tagValues += tag["tag_string_character"].split()
                tagValues += tag["tag_string_meta"].split()
                tagValues = list(set(tagValues))
                filteredTagValues = []
                for t in tagValues:
                    filtered_string = re.sub(r'[^\w\s"\']|_', " ", t)
                    filtered_string = re.sub(r'"', "'", filtered_string)
                    filtered_string = filtered_string.strip()
                    if filtered_string != "":
                        filteredTagValues.append(filtered_string)
                # random.shuffle(filteredTagValues)
                imgTag = ",".join(filteredTagValues)
                output.append({"img_src": imgFile, "id": id, "tag": imgTag})
                # output.append({'file_name': tagSearch[0], 'text': imgTag})
            else:
                print(f"Unable to find tag file with id {id}")
    else:
        if len(files_to_skip):
            duplicates = set(files_to_skip).intersection(
                set([osp.basename(f) for f in imgList])
            )
            imgList = [f for f in imgList if f not in duplicates]
            print(f"{len(duplicates)} duplicate")
        output = imgList

    for i in range(len(output) // args.split_by + 1):
        i = i + zip_start_index
        folderName = osp.join(args.output_path, str(i))
        os.makedirs(folderName, exist_ok=True)
        with open(osp.join(folderName, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    for idx, item in enumerate(tqdm(output)):
        folder_idx = (idx // args.split_by) + zip_start_index
        if type(item) == dict:
            imgFile = item["img_src"]
            tag_dst = osp.join(args.output_path, f"{str(folder_idx)}", f"{item['id']}.txt")
            with open(tag_dst, "w") as f:
                f.write(item["tag"])
        else:
            imgFile = item
        if args.screenshot:
            video_name, e = osp.split(imgFile)[0].split("/")[-2:]
            ext = osp.splitext(imgFile)[1]
            img_dst = osp.join(
                args.output_path, f"{str(folder_idx)}", f"{video_name}_{e}_{idx}{ext}"
            )
        else:
            img_dst = osp.join(args.output_path, f"{str(folder_idx)}", osp.basename(imgFile))
        if not osp.isfile(img_dst):
            os.symlink(osp.abspath(imgFile), img_dst)

    for i in range(len(output) // args.split_by + 1):
        folder_idx = i + zip_start_index
        if osp.isfile(osp.join(args.output_path, f"{folder_idx}.tar.gz")):
            print(f"{folder_idx}.tar.gz already exists, skipping...")
        else:
            print(f"start to create zip {folder_idx}.tar.gz")
            os.system(
                f"cd {args.output_path} && tar chf - {folder_idx} | pigz -p 12 > {folder_idx}.tar.gz"
            )
            print(f"start to upload zip {folder_idx}.tar.gz")
            s3.fput_object(
                args.bucket_name,
                f"{folder_idx}.tar.gz",
                osp.join(args.output_path, f"{folder_idx}.tar.gz"),
            )
    
#%%
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare image for training')
    parser.add_argument('--image_path', type=str, help='Path to the data')
    parser.add_argument('--output_path', type=str, help='Path to the output')
    parser.add_argument('--split_by', type=int, default=2000, help='Number of items to split the data into')
    parser.add_argument('--screenshot', action='store_true', help='Flag to indicate whether is screenshot')
    parser.add_argument('--source', type=str, default='', help='image source')
    parser.add_argument('--wd14_thresh', type=float, default=0.35, help='Threshold value for wd14')
    parser.add_argument('--custom_tags', type=str, default="", help='Custom tags for the data')
    parser.add_argument('--bucket_name', type=str, default='queue-1', help='Name of the bucket')
    parser.add_argument('--interactive', action='store_true', help='interactive mode')

    args = parser.parse_args()
    def input_or_default(name, default, type_func, args):
        if not args.interactive:
            return default
        while True:
            user_input = input(f"Enter a value for '{name}':[default:{default}]\n")
            if user_input == "":
                return
            try:
                value = type_func(user_input)
                setattr(args, name, value)
                return
            except ValueError:
                print(f"Invalid input, please enter a value of type {type_func.__name__}.")
                
    input_or_default("bucket_name", args.bucket_name, str, args)
    input_or_default("image_path", args.image_path, str, args)
    input_or_default("output_path", args.output_path, str, args)
    input_or_default("split_by", args.split_by, int, args)
    input_or_default("screenshot", args.screenshot, bool, args)
    input_or_default("source", args.source, str, args)
    input_or_default("wd14_thresh", args.wd14_thresh, float, args)
    input_or_default("custom_tags", args.custom_tags, str, args)
    
    assert args.image_path is not None, "image_path is required"
    assert args.output_path is not None, "output_path is required"
    
    main(args)

