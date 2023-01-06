#%%
import os
import os.path as osp
from glob import glob

data_path = "/mnt/d/data/"
dst = osp.join(data_path, "output")
os.makedirs(dst, exist_ok=True)
files = glob(osp.join(data_path, '**/*.npz'), recursive=True)
# %%
import shutil
for file in files:
    name = osp.basename(file)
    shutil.move(file, osp.join(dst,name))

#%%
meta_files = glob(osp.join(data_path, "**/meta_lat.json"), recursive=True)
# %%
import json
metas = {}
for file in meta_files:
    with open(file, 'r') as f:
        metas = {**metas, ** json.load(f)}
import re
def clean(tag):
    filtered_string = re.sub(r'[^\w\s"\']|_', ' ', tag)
    filtered_string = re.sub(r'"', "'", filtered_string)
    filtered_string = re.sub(r' +', " ", filtered_string)
    return filtered_string.strip()
# %%
filtered_meta = {k : v for k ,v in metas.items() if 'train_resolution' in v}

# %%
with open(osp.join(dst, 'meta_lat.json'),'w') as f:
    json.dump(filtered_meta, f, indent=4)
    
# %%
for k,v in filtered_meta.items():
    v['tags'] = ','.join([clean(t) for t in v['tags'].split(',')])
# %%
