import os
import sys
from pathlib import Path
import os.path as osp

# will add path to kohya trainer in settings
import logging

from ..preprocess.utils import *
from huggingface_utils import *
logger = logging.getLogger()
setup_logger(logger)
import settings
import train

def main():
    path_to_s3_data = settings.path_to_s3_data
    bucket_name = settings.bucket_name
    local_data_path = osp.join(settings.data_download_path, path_to_s3_data)
    
    s3_download(bucket_name, path_to_s3_data)
    extract(local_data_path)
    
    # clone repo to resume if repo has been created
    if model_exist(settings.hub_id):
        logger.info("Repo already exist, will clone and resume training")
        os.system(f"git clone --depth 1 https://huggingface.co/{settings.hub_id} {settings.output_dir}")
    else:
        logger.info("Repo not find, will starting training from base model")
        
    train.main(settings.model_checkpoint,
                settings.class_file,
                settings.data_dir,
                batch_size = settings.batch_size,
                max_train_samples = settings.max_train_samples,
                max_eval_samples = settings.max_eval_samples,
                seed = settings.seed,
                cache_dir = settings.cache_dir,
                eval_metric = settings.eval_metric,
                eval_threshold = settings.eval_threshold,
                output_dir = settings.output_dir,
                log_to = settings.log_to,
                lr_scheduler_type = settings.lr_scheduler_type,
                warmup_ratio = settings.warmup_ratio,
                learning_rate = settings.learning_rate,
                num_train_epochs = settings.num_train_epochs,
                weight_decay = settings.weight_decay,
                gradient_accumulation_steps = settings.gradient_accumulation_steps,
                gradient_checkpointing = settings.gradient_checkpointing,
                hub_id = settings.hub_id
    )    
    
    logger.info(f"Finish training {settings.hub_id}")
    file_to_mark_complete = "/tmp/{settings.hub_id.replace('/', '_')}"
    Path(file_to_mark_complete).touch()
    upload(settings.bucket_name, osp.basename(file_to_mark_complete), file_to_mark_complete)
    
    
    
    
    
    