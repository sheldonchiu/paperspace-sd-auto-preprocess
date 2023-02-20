import os
import sys
import os.path as osp
from glob import glob
import argparse
import settings
import logging
import gzip, tarfile
import shutil
from tqdm.auto import tqdm
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)
try:
    from minio import Minio
except:
    logger.warning('Minio was not installed')

DEFAULT_WD14_TAGGER_REPO = 'SmilingWolf/wd-v1-4-convnext-tagger-v2'

def get_s3_connection() -> Minio:
    return Minio(
            settings.s3_endpoint_url,
            access_key=settings.s3_aws_access_key_id,
            secret_key=settings.s3_aws_secret_access_key,
        )

def prepare_wd_parser(train_data_dir: str, thresh: float=0.35, batch_size: int=4, caption_extention: str='.txt') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_WD14_TAGGER_REPO,
                        help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID")
    parser.add_argument("--model_dir", type=str, default="wd14_tagger_model",
                        help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ")
    parser.add_argument("--force_download", action='store_true',
                        help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします")
    parser.add_argument("--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=None,
                        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
    parser.add_argument("--caption_extention", type=str, default=None,
                        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args([train_data_dir, 
                              '--thresh', str(thresh), 
                              '--batch_size',str(batch_size), 
                              '--max_data_loader_n_workers', '0', 
                              '--caption_extension', caption_extention,
                              '--model_dir', '/tmp/wd14_tagger_model'])
    
    return args

def prepare_caption_parser(train_data_dir: str, batch_size: int = 4, caption_extention: str = '.caption') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--caption_weights", type=str, default="https://huggingface.co/sheldonxxxx/ofa_for_repo/resolve/main/caption_huge_best.pt",
                        help="OFA caption weights (caption_huge_best.pth) / OFA captionの重みファイル(model_large_caption.pth)")
    parser.add_argument("--caption_extention", type=str, default=None,
                        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
    parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--batch_size", type=int, default=3, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=None,
                        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
    parser.add_argument("--num_beams", type=int, default=5, help="num of beams in beam search /beam search時のビーム数（多いと精度が上がるが時間がかかる）")
    parser.add_argument("--temperature", type=float, default=0.5, help="top_p in Nucleus sampling / Nucleus sampling時のtop_p")
    parser.add_argument("--max_length", type=int, default=16, help="max length of caption / captionの最大長")
    parser.add_argument("--min_length", type=int, default=5, help="min length of caption / captionの最小長")
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility / 再現性を確保するための乱数seed')
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int, help='')
    parser.add_argument("--fp16", action="store_true", help="inference with fp16")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args([train_data_dir, 
                              '--batch_size',str(batch_size), 
                              '--max_data_loader_n_workers', '0',
                              '--caption_extension', caption_extention])
    return args

def prepare_merge_parser(train_data_dir: str, out_json: str, caption_extension: str, in_json: str=None, recursive: bool=False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("--in_json", type=str,
                        help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")
    parser.add_argument("--caption_extention", type=str, default=None,
                        help="extension of caption file (for backward compatibility) / 読み込むキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
    parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 読み込むキャプションファイルの拡張子")
    parser.add_argument("--full_path", action="store_true",
                        help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）")
    parser.add_argument("--recursive", action="store_true",
                        help="recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    params = [train_data_dir, out_json, '--caption_extension', caption_extension]
    if recursive:
        params += ['--recursive']
    if in_json:
        params += ['--in_json', in_json]
    args = parser.parse_args(params)
    return args

def prepare_clean_parser(in_json: str,out_json: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("in_json", type=str, help="metadata file to input / 読み込むメタデータファイル")
    parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args([in_json, out_json])
    return args

def prepare_bucket_parser(train_data_dir: str, in_json: str, out_json: str,
                          model_name_or_path: str, upscale_model_dir: str,
                          debug_dir: str=None, upscale_outscale: int=None, 
                          batch_size: int=None, flip_aug: bool=False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str,
                        help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("in_json", type=str,
                        help="metadata file to input / 読み込むメタデータファイル")
    parser.add_argument("out_json", type=str,
                        help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("model_name_or_path", type=str,
                        help="model name or path to encode latents / latentを取得するためのモデル")
    parser.add_argument("--v2", action='store_true',
                        help='not used (for backward compatibility) / 使用されません（互換性のため残してあります）')
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=None,
                    help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
    parser.add_argument("--max_resolution", type=str, default="768,768",
                        help="max resolution in fine tuning (width,height) / fine tuning時の最大画像サイズ 「幅,高さ」（使用メモリ量に関係します）")
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--bucket_reso_steps", type=int, default=64,
                        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します")
    parser.add_argument("--bucket_no_upscale", action="store_true",
                        help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
    parser.add_argument("--full_path", action="store_true",
                        help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）")
    parser.add_argument("--flip_aug", action="store_true",
                        help="flip augmentation, save latents for flipped images / 左右反転した画像もlatentを取得、保存する")
    parser.add_argument("--upscale", action="store_true",
                        help="upscale before resize")
    parser.add_argument(
        '--upscale_model_name',
        type=str,
        default='RealESRGAN_x4plus_anime_6B',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
                'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('--upscale_outscale', type=int, default=2,
                        help='')
    parser.add_argument(
        '--upscale_denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
            'Only used for the realesr-general-x4v3 model'))
    parser.add_argument(
        '--upscale_model_dir', type=str, default='upscale', help='[Option] Model path.')
    parser.add_argument('--upscale_tile', type=int, default=512,
                        help='Tile size, 0 for no tile during testing')
    parser.add_argument('--upscale_tile_pad', type=int,
                        default=10, help='Tile padding')
    parser.add_argument('--upscale_pre_pad', type=int,
                        default=0, help='Pre padding size at each border')
    parser.add_argument("--upscale_enable_reso", type=int, default=1000*1000,
                        help="Images with resolution(w*h) below this will upscale before resize, if upsacle is enabled")
    parser.add_argument(
        '--debug_dir', type=str, default=None, help='')  
    parser.add_argument("--skip_existing", action="store_true",
                    help="skip images if npz already exists (both normal and flipped exists if flip_aug is enabled) / npzが既に存在する画像をスキップする（flip_aug有効時は通常、反転の両方が存在する画像をスキップ）")

    s = [   train_data_dir, 
            in_json,out_json,
            model_name_or_path,
            '--skip_existing',
        ]
    if upscale_model_dir:
        s+= ['--upscale', '--upscale_model_dir', upscale_model_dir]
    if flip_aug:
        s += ['--flip_aug']
    if upscale_outscale:
        s += ['--upscale_outscale', str(upscale_outscale)]
    if debug_dir:
        s += ['--debug_dir', debug_dir]
    if batch_size:
        s += ['--batch_size', str(batch_size)]
    args = parser.parse_args(s)

    return args

def download_model(model_dir, url):
    local_path = osp.join(model_dir, osp.basename(url))
    os.makedirs(model_dir, exist_ok=True)
    os.system(f'wget -c "{url}" -O {local_path}')
    return local_path

def upload(bucketName, remotePath, file):
    if osp.isfile(file):
        try:
            s3 = get_s3_connection()
            s3.fput_object(bucketName, remotePath, file)
            return True
        except:
            logger.exception("message")
            return False
    return False

def check_work_queue():
    data_dir = settings.data_download_path
    files = glob(osp.join(data_dir, "*.tar.gz"))
    files_to_process = []
    for f in files:
        filename = f[:f.find('.')]
        if 'result' not in f and f'{filename}-result.tar.gz' not in files:
            files_to_process.append(f)
    return len(files_to_process)

def s3_download(bucketName, remotePath, localPath):
    try:
        s3 = get_s3_connection()
        logger.info(f"Downloading {remotePath}")
        s3.fget_object(bucketName, remotePath, localPath)
        logger.info(f"Finish Downloading {remotePath}")
        return True
    except:
        logger.exception(f"Download failed for file {remotePath}")
        if osp.isfile(localPath):
            os.remove(localPath)
        return False

def download_with_queue(data):
    bucketName, remotePath, localPath = data
    if settings.skip_download or osp.isfile(localPath):
        return True
    while True:
        queue_size = check_work_queue()
        if queue_size > settings.prefetch_num_file:
            logger.debug(f"Current number of files in queue is {queue_size}, will go to sleep")
            time.sleep(120)
        else:
            break
    return s3_download(bucketName, remotePath, localPath)
    
def get_list_of_files(bucketName):
    try:
        s3 = get_s3_connection()
        response = [o.object_name for o in s3.list_objects(bucketName)]
    except KeyError:
        response = []
    except:
        logger.error("Unable to list objects in bucket, please check the s3 storage")
    return response


def compress(members, output_filename):
    command = f"tar chf - {' '.join(members)} | pigz -p {settings.pigz_num_workers} > {output_filename}"
    logger.debug(command)
    os.system(command)

def extract(file):
    target_dir = file[:file.find('.')]
    complete_mark = target_dir + ".complete"
    if settings.skip_extract or osp.isfile(complete_mark):
        return target_dir
    
    logger.info(f"Start to extract {file}")
    with gzip.open(file, 'rb') as f_in:
        new_file = file.replace('.gz','')
        with open(new_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with tarfile.open(new_file) as f:
        f.extractall(target_dir)
        
    Path(complete_mark).touch()
    os.remove(new_file)
    return target_dir

def flatten_folder(folder: str) -> None:
    for file_path in Path(folder).rglob('*'):
        if file_path.is_file():
            new_file_path = Path(folder) / file_path.name
            shutil.move(file_path, new_file_path)    

def compress_and_upload(members, output_filename, bucketName):
    target_file = osp.basename(output_filename)
    logger.info(f"Start compression for {target_file}")
    compress(members, output_filename)
    logger.info(f"Finish compression for {target_file}")
    
    logger.info(f"Start upload for {target_file}")
    upload(bucketName, target_file, output_filename)
    logger.info(f"Finish upload for {target_file}")
    
def test_tag_extension(target_dir, tag_extension, tag_extension2):
    if len(glob(osp.join(target_dir, f"*{tag_extension}"))) > 0:
        return tag_extension
    elif len(glob(osp.join(target_dir, f"*{tag_extension2}"))) > 0:
        return tag_extension2
    else:
        return None
    
def setup_logger(logger):
    from discord_logging.handler import DiscordHandler
    # Silence requests and discord_webhook internals as otherwise this example will be too noisy
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("discord_webhook").setLevel(logging.FATAL)  # discord_webhook.webhook - ERROR - Webhook rate limited: sleeping for 0.235 seconds...
    
    stream_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    discord_format = logging.Formatter("%(message)s")
    
    discord_handler = DiscordHandler("Paperspace preprocess", settings.discord_webhook_url)
    discord_handler.setFormatter(discord_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_format)

    # Add the handlers to the Logger
    logger.addHandler(discord_handler)
    logger.addHandler(stream_handler)
    if settings.log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif settings.log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif settings.log_level == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif settings.log_level == 'ERROR':
        logger.setLevel(logging.ERROR)
    
def load_config_from_file(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        # best effort to load the custom value
        original_value = getattr(settings, key, None)
        if original_value and type(value) != type(original_value):
            if type(original_value) == bool:
                value = settings.bool_t(value)
            else:
                value = type(original_value)(value)
        setattr(settings, key, value)
        
class jobContext(dict):
    def __init__(self, **kwargs):
        self.job_name = kwargs.get('job_name')
        self.file = kwargs.get('file')

    def __enter__(self):
        logger.info(f"[START] Stage: {self.job_name} File: {self.file}")
        return self

    def __exit__(self, *exc):
        logger.info(f"[END] Stage: {self.job_name} File: {self.file}") 
        return False
    
def cache_progress_watcher(bucket_name, target_dir, key, file_extension, recursive=False, interval=60*5):
    tmp_dir = osp.join(settings.data_download_path, 'cache')
    os.makedirs(tmp_dir, exist_ok=True)
    cache_file_name = osp.join(tmp_dir, f"{key}.tar")
    complete_mark = osp.join(target_dir, 'complete')
    record = []
    while not osp.isfile(complete_mark):
        files = [f for f in glob(osp.join(target_dir, f"**/*{file_extension}" if recursive else f"*{file_extension}"), recursive=recursive) if f not in record]
        with tarfile.open(cache_file_name, 'a', dereference=True) as f:
            for file in files:
                f.add(file, arcname=osp.basename(file))
        upload(bucket_name, osp.basename(cache_file_name), cache_file_name)
        time.sleep(interval)
    
    os.remove(complete_mark)

image_format = ['.jpeg','.jpg', '.png', '.webp']
def remove_old_files(folder):
    file_format = ['.npz','.tag', '.caption']
    for file_path in Path(folder).rglob('*'):
        ext = Path(file_path).suffix
        if ext in file_format:
            os.remove(file_path)
    
def remove_images_from_folder(folder):
    images = []
    for i in image_format:
        images += glob(osp.join(folder, f"*{i}"))
    for image in images:
        os.remove(image)
        
# config = {"use_original_tags": "0"}
# with open("config.json", 'w') as f:
#     json.dump(config, f)