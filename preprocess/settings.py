import sys
import os
from os import path as osp

root_dir = os.environ.get('ROOT_DIR', '/tmp')
path_to_repo = f"{root_dir}/kohya-trainer-paperspace"
if osp.join(path_to_repo,'finetune') not in sys.path:
    sys.path.append(osp.join(path_to_repo,'finetune'))

data_download_path = '/tmp/data/'
model_path = '/tmp'
os.makedirs(data_download_path, exist_ok=True)

bool_t = lambda x: x.lower() in ['true', 'yes', '1']

use_result_as_input = bool_t(os.environ.get('USE_RESULT_AS_INPUT', '0'))
target_complete_suffix = os.environ.get('TARGET_COMPLETE_SUFFIX', 'result')

complete_suffix = os.environ.get('COMPLETE_SUFFIX', 'result')
skip_download = bool_t(os.environ.get('SKIP_DOWNLOAD', '0'))
skip_extract = bool_t(os.environ.get('SKIP_EXTRACT', '0'))
tag_using_wd14 = bool_t(os.environ.get('TAG_USING_WD14', '1'))
wd14_thresh = float(os.environ.get('WD14_THRESHOLD', 0.35))
wd14_batch_size = int(os.environ.get('WD14_BATCH_SIZE', 4))
enable_caption = bool_t(os.environ.get('ENABLE_CAPTION', '1'))
caption_type = os.environ.get('CAPTION_TYPE', 'ofa')
# kosmos2_env_path = os.environ.get('KOSMOS2_ENV_PATH', '/tmp/kosmos2-env')
# kosmos2_script_path = os.environ.get('KOSMOS2_SCRIPT_PATH', '/storage/unilm/kosmos-2/demo')
caption_batch_size = int(os.environ.get('CAPTION_BATCH_SIZE', 16))
use_original_tags = bool_t(os.environ.get('USE_ORIGINAL_TAGS', '0'))
enable_filter = bool_t(os.environ.get('ENABLE_FILTER', '1'))
filter_using_cafe_aesthetic = bool_t(os.environ.get('FILTER_USING_CAFE', '1'))
save_img_for_debug = bool_t(os.environ.get('SAVE_IMG_FOR_DEBUG', '0'))
save_original_img = bool_t(os.environ.get('SAVE_ORIGINAL_IMG','1'))

bucketing_resolution=os.environ.get('BUCKETING_RESOLUTION', '1024,1024')
bucketing_mixed_precision = os.environ.get('BUCKETING_MIXED_PRECISION', 'fp16')
bucketing_batch_szie = int(os.environ.get('BUCKETING_BATCH_SIZE', 2))
bucketing_flip_aug = bool_t(os.environ.get('BUCKETING_FLIP_AUG', '0'))
bucketing_reso_steps = int(os.environ.get('BUCKETING_RESOLUTION_STEPS', '64'))

pigz_num_workers = int(os.environ.get('PIGZ_NUM_WORKERS', 8))
prefetch_num_file = int(os.environ.get('PREFETCH_NUM_FILE', 1))

cafe_batch_size = int(os.environ.get('CAFE_BATCH_SIZE', 250))
# filter_aesthetic_thresh = float(os.environ.get('FILTER_ANORMAL_THRESH', 0.6))
filter_anime_thresh = float(os.environ.get('FILTER_ANI_THRESH', 0.6))
filter_waifu_thresh = float(os.environ.get('FILTER_WAIFU_THRESH', 0.6))

main_loop_interval = int(os.environ.get('MAIN_LOOP_INTERVAL', 60*5))

vae_model_url = os.environ.get("VAE_MODEL_URL", "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors")
vae_model_hub = os.environ.get("VAE_MODEL_HUB",None)

s3_endpoint_url = os.environ.get('S3_HOST_URL')
s3_aws_access_key_id = os.environ.get('S3_ACCESS_KEY')
s3_aws_secret_access_key = os.environ.get('S3_SECRET_KEY')
s3_bucket_name = os.environ.get('S3_BUCKET_NAME')
s3_cache_bucket_name = os.environ.get('S3_CACHE_BUCKET_NAME')

discord_webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
log_level = os.environ.get('LOG_LEVEL', 'INFO')