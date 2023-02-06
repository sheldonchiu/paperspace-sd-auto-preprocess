#%%
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os.path as osp
from glob import glob
from itertools import chain
from PIL import Image
from tqdm import tqdm
from accelerate import dispatch_model
from accelerate import infer_auto_device_map
# %%
batch_size = 2
data_path = "/data/data/images"
image_format = ['jpeg', 'jpg', 'png', 'webp']
image_paths = list(chain(
    *[glob(osp.join(data_path, f"**/*.{f}"), recursive=True) for f in image_format]))
# image_paths = glob(osp.join(data_path, "*"))
print(f"found {len(image_paths)} images.")
#%%
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
)
device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"])
model = dispatch_model(model, device_map=device_map, offload_dir="/app/huggingface_cache/offload")
#%%
def run_batch(path_imgs):
    imgs = torch.stack([im for _, im in path_imgs]).to(device)
    with torch.no_grad():
        captions = model.generate({"image": imgs}, num_beams=3, temperature=0.7,repetition_penalty=1.5,max_length=72)
    for (image_path, _), caption in zip(path_imgs, captions):
        with open(osp.splitext(image_path)[0] + ".caption2", "wt", encoding='utf-8') as f:
            f.write(caption + "\n")
    
b_imgs = []
for image_path in tqdm(image_paths, smoothing=0.0):
    raw_image = Image.open(image_path)
    if raw_image.mode != "RGB":
        print(f"convert image mode {raw_image.mode} to RGB: {image_path}")
        raw_image = raw_image.convert("RGB")   
    image = vis_processors["eval"](raw_image)
    b_imgs.append((image_path, image))
    if len(b_imgs) >= batch_size:
        run_batch(b_imgs)
        b_imgs.clear()
if len(b_imgs) > 0:
    run_batch(b_imgs)
#%%
# torch.stack()
# # image = image_paths[6]
# # caption = image_caption(Image.open(image))
# # print(caption)
# for image in tqdm(image_paths):
#     dst_file = osp.join(osp.dirname(image), osp.splitext(osp.basename(image))[0]+".caption2")
#     if osp.exists(dst_file):
#         continue
#     caption = image_caption(Image.open(image))
#     # print(caption)
#     with open(dst_file, 'w') as f:
#         f.write(caption)
        




#export HUGGINGFACE_HUB_CACHE=/app/huggingface_cache
# export TORCH_HOME=/app/huggingface_cache