#%%
from dotenv import load_dotenv
load_dotenv()

import os
import os.path as osp
import json
from glob import glob
import click
from tqdm import tqdm
from utils import MinioClient

s3_client = MinioClient(
    os.environ["S3_HOST_URL"],
    os.environ["S3_ACCESS_KEY"],
    os.environ["S3_SECRET_KEY"],
    secure=False
)

@click.command()
@click.option('--output_path', type=str, help='Path to the output', prompt='Enter the output path:')
@click.option('--bucket_name', type=str, default='queue-1', help='Name of the bucket', prompt='Enter the bucket name:')
@click.option('--start_idx', type=int, default=0, help='Start downloading from this index', prompt='Enter the start index:')
@click.option('--end_idx', type=int, default=-1, help='End downloading from this index', prompt='Enter the end index:')
@click.option('--download', is_flag=True, default=True, help='Merge files', prompt='Perform download?')
@click.option('--merge', is_flag=True, default=True, help='Merge files', prompt='Enable file merging?')
@click.option('--npz_only', is_flag=True, help='Keep images')
def main(output_path, bucket_name, start_idx, end_idx, download, merge, npz_only):
    if download:
        history = []
        os.makedirs(output_path, exist_ok=True)
        history_file = osp.join(output_path, "done.json")
        if osp.isfile(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
        
        list_of_files = [o.object_name for o in s3_client.list_objects(bucket_name) if o.object_name not in history]
        if start_idx > 0:
            target_files = [f"{i}-result.tar.gz" for i in range(0, start_idx)]
            list_of_files = [f for f in list_of_files if f not in target_files]
        if end_idx != -1:
            target_files = [f"{i}-result.tar.gz" for i in range(start_idx, end_idx+1)]
            list_of_files = [f for f in list_of_files if f in target_files]
        print(f"Found {len(list_of_files)} files")
        
        for file in tqdm(list_of_files):
            dst_path = osp.join(output_path, file)
            s3_client.download_file(bucket_name, file, dst_path)
            os.system(f"pigz -dc {dst_path} | tar xf - -C {output_path}")
            os.remove(dst_path)
            history.append(file)
            with open(history_file, "w") as f:
                json.dump(history, f)  
        
    if merge:
        target_path = osp.join(output_path, "merge")
        os.makedirs(target_path, exist_ok=True)
        if npz_only:
            files = [f for f in glob(osp.join(output_path, "**/*.npz"), recursive=True)]
        else:
            files = [f for f in glob(osp.join(output_path, "tmp/data/**/*"), recursive=True)]
        
        for file in tqdm(files):          
            try:
                if 'json' not in osp.basename(file):
                    os.symlink(file, osp.join(target_path, osp.basename(file)))
            except FileExistsError:
                print(f"{osp.basename(file)} alreay exist")
                
        data = {}
        metas = glob(osp.join(output_path, "**/meta_lat.json"), recursive=True)
        for meta in metas:
            with open(meta, 'r') as f:
                data = data | json.load(f)
        with open(osp.join(target_path, "meta_lat.json"), 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
