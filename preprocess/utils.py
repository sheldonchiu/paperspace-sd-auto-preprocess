import os
import sys
import os.path as osp
from minio import Minio
from minio.error import S3Error
from glob import glob
import argparse
import settings
import logging
from discord_logging.handler import DiscordHandler
import gzip, tarfile
import shutil
from tqdm.auto import tqdm
import time
from pathlib import Path

import re
logger = logging.getLogger(__name__)

s3 = Minio(
    settings.s3_endpoint_url,
    access_key=settings.s3_aws_access_key_id,
    secret_key=settings.s3_aws_secret_access_key,
)

WD14_TAGGER_REPO = 'SmilingWolf/wd-v1-4-vit-tagger'

def prepare_wd_parser(train_data_dir, thresh=0.35, batch_size=4, caption_extention='.txt'):
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--repo_id", type=str, default=WD14_TAGGER_REPO,
                        help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID")
    parser.add_argument("--model_dir", type=str, default="wd14_tagger_model",
                        help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ")
    parser.add_argument("--force_download", action='store_true',
                        help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします")
    parser.add_argument("--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--caption_extention", type=str, default=None,
                        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args([train_data_dir, 
                              '--thresh', str(thresh), 
                              '--batch_size',str(batch_size), 
                              '--caption_extension', caption_extention,
                              '--model_dir', '/tmp/wd14_tagger_model'])
    
    return args

def prepare_caption_parser(train_data_dir, batch_size=4, caption_extention='.caption'):
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--caption_weights", type=str, default="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth",
                        help="BLIP caption weights (model_large_caption.pth) / BLIP captionの重みファイル(model_large_caption.pth)")
    parser.add_argument("--caption_extention", type=str, default=None,
                        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
    parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--beam_search", action="store_true",
                        help="use beam search (default Nucleus sampling) / beam searchを使う（このオプション未指定時はNucleus sampling）")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--num_beams", type=int, default=1, help="num of beams in beam search /beam search時のビーム数（多いと精度が上がるが時間がかかる）")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p in Nucleus sampling / Nucleus sampling時のtop_p")
    parser.add_argument("--max_length", type=int, default=75, help="max length of caption / captionの最大長")
    parser.add_argument("--min_length", type=int, default=5, help="min length of caption / captionの最小長")
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility / 再現性を確保するための乱数seed')
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args([train_data_dir, 
                              '--batch_size',str(batch_size), 
                              '--caption_extension', caption_extention])
    return args

def prepare_merge_parser(train_data_dir, out_json, caption_extension, in_json=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("--in_json", type=str, help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")
    parser.add_argument("--full_path", action="store_true",
                        help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）")
    parser.add_argument("--debug", action="store_true", help="debug mode, print tags")
    parser.add_argument("--caption_extension", type=str, default='.txt', help="")
    params = [train_data_dir, out_json, '--caption_extension', caption_extension]
    if in_json:
        params += ['--in_json', in_json]
    args = parser.parse_args(params)
    return args

def prepare_clean_parser(in_json,out_json):
    parser = argparse.ArgumentParser()
    # parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("in_json", type=str, help="metadata file to input / 読み込むメタデータファイル")
    parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")

    args = parser.parse_args([in_json, out_json])
    return args

def prepare_bucket_parser(train_data_dir, in_json, out_json,
                          model_name_or_path, upscale_model_dir, 
                          model_name_or_path_v2=None,
                          debug_dir=None, upscale_outscale=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str,
                        help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("in_json", type=str,
                        help="metadata file to input / 読み込むメタデータファイル")
    parser.add_argument("out_json", type=str,
                        help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("model_name_or_path", type=str,
                        help="model name or path to encode latents / latentを取得するためのモデル")
    parser.add_argument("--model_name_or_path_v2", type=str, default=None,
                        help="model name or path to encode latents / latentを取得するためのモデル")
    parser.add_argument("--v2", action='store_true',
                        help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む')
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--max_resolution", type=str, default="768,768",
                        help="max resolution in fine tuning (width,height) / fine tuning時の最大画像サイズ 「幅,高さ」（使用メモリ量に関係します）")
    parser.add_argument("--min_bucket_reso", type=int, default=256,
                        help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024,
                        help="maximum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
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
    s = [   train_data_dir, 
            in_json,out_json,
            model_name_or_path,
            '--flip_aug',
            '--upscale',
            '--upscale_model_dir', upscale_model_dir
        ]
    if upscale_outscale:
        s += ['--upscale_outscale', str(upscale_outscale)]
    if debug_dir:
        s += ['--debug_dir', debug_dir]
    if model_name_or_path_v2:
        s += ['--model_name_or_path_v2', model_name_or_path_v2]
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

def download(data):
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
    try:
        logger.info(f"Downloading {remotePath}")
        s3.fget_object(bucketName, remotePath, localPath)
        logger.info(f"Finish Downloading {remotePath}")
        return True
    except:
        logger.exception("message")
        return False
    
def get_list_of_files(bucketName):
    try:
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
        f.extractall(settings.data_download_path)
    os.remove(new_file)
    Path(complete_mark).touch()
    return target_dir

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
    logger.setLevel(logging.INFO)