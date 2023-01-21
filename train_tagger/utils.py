import os
import sys
import os.path as osp
from minio import Minio
import settings
import logging
from discord_logging.handler import DiscordHandler
import gzip, tarfile
import shutil
from tqdm.auto import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)

s3 = Minio(
    settings.s3_endpoint_url,
    access_key=settings.s3_aws_access_key_id,
    secret_key=settings.s3_aws_secret_access_key,
)

def upload(bucketName, remotePath, file):
    if osp.isfile(file):
        try:
            s3.fput_object(bucketName, remotePath, file)
            return True
        except:
            logger.exception("message")
            return False
    return False

def s3_download(bucketName, remotePath, localPath):
    try:
        logger.info(f"Downloading {remotePath}")
        s3.fget_object(bucketName, remotePath, localPath)
        logger.info(f"Finish Downloading {remotePath}")
        return True
    except:
        logger.exception("message")
        return False

def extract(file):
    target_dir = file[:file.find('.')]
    
    logger.info(f"Start to extract {file}")
    with gzip.open(file, 'rb') as f_in:
        new_file = file.replace('.gz','')
        with open(new_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with tarfile.open(new_file) as f:
        f.extractall(settings.data_download_path)
    os.remove(new_file)
    return target_dir
    
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
        
        
# config = {"use_original_tags": "0"}
# with open("config.json", 'w') as f:
#     json.dump(config, f)