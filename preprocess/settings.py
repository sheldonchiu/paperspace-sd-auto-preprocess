import sys
import os
from os import path as osp

path_to_repo = "/notebooks/kohya-trainer-paperspace"
if osp.join(path_to_repo,'finetune') not in sys.path:
    sys.path.append(osp.join(path_to_repo,'finetune'))

data_download_path = '/tmp/data/'
model_path = '/tmp'
os.makedirs(data_download_path, exist_ok=True)

bool_t = lambda x: x.lower() in ['true', 'yes', '1']

skip_download = bool_t(os.environ.get('SKIP_DOWNLOAD', '0'))
skip_extract = bool_t(os.environ.get('SKIP_EXTRACT', '0'))
tag_using_wd14 = bool_t(os.environ.get('TAG_USING_WD14', '1'))
wd14_batch_size = int(os.environ.get('WD14_BATCH_SIZE', 4))
caption_using_blip = bool_t(os.environ.get('CAPTION_USING_BLIP', '1'))
blip_batch_size = int(os.environ.get('BLIP_BATCH_SIZE', 16))
use_original_tags = bool_t(os.environ.get('USE_ORIGINAL_TAGS', '0'))
enable_filter = bool_t(os.environ.get('ENABLE_FILTER', '1'))
filter_using_cafe_aesthetic = bool_t(os.environ.get('FILTER_USING_CAFE', '0'))
save_img_for_debug = bool_t(os.environ.get('SAVE_IMG_FOR_DEBUG', '1'))
upscale_outscale = int(os.environ.get('UPSCALE_OUTSCALE', 2))
pigz_num_workers = int(os.environ.get('PIGZ_NUM_WORKERS', 8))
prefetch_num_file = int(os.environ.get('PREFETCH_NUM_FILE', 1))

cafe_batch_size = int(os.environ.get('CAFE_BATCH_SIZE', 250))
filter_aesthetic_thresh = float(os.environ.get('FILTER_ANORMAL_THRESH', 0.6))
filter_anime_thresh = float(os.environ.get('FILTER_ANI_THRESH', 0.6))
filter_waifu_thresh = float(os.environ.get('FILTER_WAIFU_THRESH', 0.6))

vae_model_url = os.environ.get("VAE_MODEL_URL", "https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float32.ckpt")
vae_model_hub = os.environ.get("VAE_MODEL_HUB",None)

vae_model_url_2 = os.environ.get("VAE_MODEL_URL_2", None)
vae_model_hub_2 = os.environ.get("VAE_MODEL_HUB_2","stabilityai/stable-diffusion-2")

s3_endpoint_url = os.environ['S3_HOST_URL']
s3_aws_access_key_id = os.environ['S3_ACCESS_KEY']
s3_aws_secret_access_key = os.environ['S3_SECRET_KEY']

discord_webhook_url = os.environ['DISCORD_WEBHOOK_URL']