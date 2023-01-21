import sys
import os
from os import path as osp

data_download_path = '/tmp/data/'
model_path = '/tmp'
os.makedirs(data_download_path, exist_ok=True)

bool_t = lambda x: x.lower() in ['true', 'yes', '1']

model_checkpoint = os.environ.get('MODEL_CHECKPOINT', 'microsoft/beit-base-patch16-384')
class_file = osp.join(data_download_path, os.environ.get('CLASS_FILE', "class.txt"))
data_dir = osp.join(data_download_path,os.environ.get('DATA_DIR', 'data'))
batch_size = int(os.environ.get('BATCH_SIZE', '96'))

max_train_samples = os.environ.get('MAX_TRAIN_SAMPLES', None)
max_train_samples = int(max_train_samples) if max_train_samples is not None else max_train_samples
max_eval_samples = os.environ.get('MAX_EVAL_SAMPLES', None)
max_eval_samples = int(max_eval_samples) if max_eval_samples is not None else max_eval_samples

seed = int(os.environ.get('SEED', 42))
cache_dir = os.environ.get('CACHE_DIR', None)
eval_metric = os.environ.get('EVAL_METRIC', 'f1')
eval_threshold = float(os.environ.get('EVAL_THRESHOLD', 0.6))
output_dir = osp.join(model_path, os.environ.get('OUTPUT_DIR', 'mecha-tagger'))
log_to = os.environ.get('LOG_TO', 'wandb').split(',')
lr_scheduler_type = os.environ.get('LR_SCHEDULER_TYPE', 'constant')
warmup_ratio = float(os.environ.get('WARMUP_RATIO', 0.05))
learning_rate = float(os.environ.get('LEARNING_RATE', 2e-5))

num_train_epochs = int(os.environ.get('NUM_TRAIN_EPOCHS', 50))
weight_decay = float(os.environ.get('WEIGHT_DECAY', 0.3))
gradient_accumulation_steps = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
gradient_checkpointing = bool_t(os.environ.get('GRADIENT_CHECKPOINTING', '1'))
hub_id = os.environ.get('HUB_ID', None)

s3_endpoint_url = os.environ['S3_HOST_URL']
s3_aws_access_key_id = os.environ['S3_ACCESS_KEY']
s3_aws_secret_access_key = os.environ['S3_SECRET_KEY']
hf_token = os.environ['HF_TOKEN']
path_to_s3_data = os.environ['PATH_TO_S3_DATA']
bucket_name = os.environ['S3_BUCKET_NAME']

discord_webhook_url = os.environ['DISCORD_WEBHOOK_URL']