#%%
import os
import os.path as osp
from glob import glob
import json
from tqdm import tqdm
import argparse
from utils import MinioClient, input_or_default

s3_client = MinioClient(
    os.environ["S3_HOST_URL"],
    os.environ["S3_ACCESS_KEY"],
    os.environ["S3_SECRET_KEY"],
    secure=False
)

def main(args):
    history = []
    os.makedirs(args.output_path, exist_ok=True)
    history_file = osp.join(args.output_path, "done.json")
    if osp.isfile(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    
    list_of_files = [o.object_name for o in s3_client.list_objects(args.bucket_name) if o.object_name not in history]
    if args.start_idx > 0:
        target_files = [f"{i}-result.tar.gz" for i in range(0, args.start_idx)]
        list_of_files = [f for f in list_of_files if f not in target_files]
    if args.end_idx != -1:
        target_files = [f"{i}-result.tar.gz" for i in range(args.start_idx, args.end_idx+1)]
        list_of_files = [f for f in list_of_files if f in target_files]
    print(f"Found {len(list_of_files)} files")
    
    for file in tqdm(list_of_files):
        dst_path = osp.join(args.output_path, file)
        s3_client.download_file(args.bucket_name, file, dst_path)
        os.system(f"pigz -dc {dst_path} | tar xf - -C {args.output_path}")
        os.remove(dst_path)
        history.append(file)
        with open(history_file, "w") as f:
            json.dump(history, f)  
        
    if args.merge:
        target_path = osp.join(args.output_path, "merge")
        os.makedirs(target_path, exist_ok=True)
        if args.npz_only:
            files = [f for f in glob(osp.join(args.output_path, "**/*.npz"), recursive=True)]
        else:
            files = [f for f in glob(osp.join(args.output_path, "tmp/data/**/*"), recursive=True)]
        
        for file in tqdm(files):          
            try:
                if 'json' not in osp.basename(file):
                    os.symlink(file, osp.join(target_path, osp.basename(file)))
            except FileExistsError:
                print(f"{osp.basename(file)} alreay exist")
                
        data = {}
        metas = glob(osp.join(args.output_path, "**/meta_lat.json"), recursive=True)
        for meta in metas:
            with open(meta, 'r') as f:
                data = data | json.load(f)
        with open(osp.join(target_path, "meta_lat.json"), 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare image for training')
    parser.add_argument('--output_path', type=str, help='Path to the output')
    parser.add_argument('--bucket_name', type=str, default='queue-1', help='Name of the bucket')
    parser.add_argument('--start_idx', type=int, default=0, help='Start downloading from this index')
    parser.add_argument('--end_idx', type=int, default=-1, help='End downloading from this index')
    parser.add_argument('--merge', action='store_true', help='merge files')
    parser.add_argument('--npz_only', action='store_true', help='keep images')
    parser.add_argument('--interactive', action='store_true', help='interactive mode')
    args = parser.parse_args()
                
    input_or_default("bucket_name", args.bucket_name, str, args)
    input_or_default("output_path", args.output_path, str, args)
    input_or_default("start_idx", args.start_idx, int, args)
    input_or_default("end_idx", args.end_idx, int, args)
    input_or_default("merge", args.merge, bool, args)
    input_or_default("npz_only", args.npz_only, bool, args)
    
    assert args.output_path is not None, "output_path is required"
    assert args.start_idx > 0, "start_idx must be greater than 0"
    assert args.start_idx <= args.end_idx, "start_idx must be less than end_idx"
    
    main(args)
