import os
import sys
import time
import os.path as osp
from pathlib import Path

# will add path to kohya trainer in settings
import logging
from utils import *
logger = logging.getLogger()
setup_logger(logger)

import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import active_children
import signal
import settings
import importlib

import file_filter
# import make_captions_by_ofa
import tag_images_by_wd14_tagger
import merge_dd_tags_to_metadata
import merge_captions_to_metadata
import clean_captions_and_tags
import prepare_buckets_latents

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    active = active_children()
    for child in active:
        child.kill()
    sys.exit(0)
    
def main():
    signal.signal(signal.SIGINT, sigterm_handler)
    downloader = ProcessPoolExecutor(max_workers=1)
    uploader = ProcessPoolExecutor(max_workers=2) # low chance of process randomly dead, set to 2 to avoid stuck for upload
    # fork will cause tf cudu init error, unknown reason
    context = multiprocessing.get_context('spawn')
    bucket_name = settings.s3_bucket_name
    files = get_list_of_files(bucket_name)
    files_to_process = []
    for f in files:
        filename = f[:f.find('.')]
        if settings.use_result_as_input:
            # if use previous result as input, 1. check if suffix matches 2. if is complete, skip
            if f.endswith(f"-{settings.target_complete_suffix}.tar.gz") and f'{filename}-{settings.complete_suffix}.tar.gz' not in files:
                files_to_process.append(f)
        elif settings.complete_suffix not in f and f'{filename}-{settings.complete_suffix}.tar.gz' not in files:
            files_to_process.append(f)
            
    results = downloader.map(download_with_queue, [(bucket_name, f, osp.join(settings.data_download_path, osp.basename(f))) for f in files_to_process])

    if settings.vae_model_url:
        # sd_model_path = osp.join(settings.model_path, 'stable-diffusion', osp.basename(settings.vae_model_url))
        sd_model_path = download_model(osp.join(settings.model_path, 'stable-diffusion'), settings.vae_model_url)
    else:
        sd_model_path = settings.vae_model_hub

    for file in files_to_process:    
        # cache_result = None
        tag_extension = '.tag'
        caption_extension = '.caption'
        meta_file = None
        local_path = osp.join(settings.data_download_path, osp.basename(file))
        if next(results) and (target_dir := extract(local_path)):
            try:
                # move all files inside target_dir to top level for easier processing
                flatten_folder(target_dir)
                if settings.use_result_as_input:
                    remove_old_files(target_dir)
                # reload setting to clean previous custom settings
                importlib.reload(settings)
                # load custom settings from config file in job zip
                config_file_path = osp.join(target_dir, "config.json")
                if osp.exists(config_file_path):
                    load_config_from_file(config_file_path)
                
                filter_dst = f"{target_dir}_filter" if settings.enable_filter else target_dir
                debug_dir = f"{target_dir}_debug" if settings.save_img_for_debug else None
                
                if debug_dir is not None:
                    os.makedirs(debug_dir, exist_ok=True)
                    
                if  settings.tag_using_wd14:
                    with jobContext(job_name="tag", file=file):
                    # create tag using wd14 and storing with .tag extension in the same directory
                        wd_args = prepare_wd_parser(target_dir, thresh=settings.wd14_thresh, batch_size=settings.wd14_batch_size, caption_extention=tag_extension)
                        task = context.Process(target=tag_images_by_wd14_tagger.main, args=(wd_args,))
                        task.start(); task.join()
                
                if settings.enable_filter:
                    with jobContext(job_name="filter",file=file):
                        # use tag created by wd14 and filter, save symbolic links in folder {train_dir}_filter
                        # since data dir has changed, need to update target_dir
                        task = context.Process(target=file_filter.main, args=(target_dir, filter_dst, tag_extension, caption_extension, settings.filter_using_cafe_aesthetic, debug_dir, config_file_path))
                        task.start(); task.join()
                    
                if settings.enable_caption:
                    with jobContext(job_name="caption", file=file):
                        # if settings.caption_type == 'ofa':
                        #     caption_args = prepare_caption_parser(filter_dst, batch_size=settings.caption_batch_size, caption_extention=caption_extension)
                        #     task = context.Process(target=make_captions_by_ofa.main, args=(caption_args,))
                        #     task.start(); task.join()
                        # elif settings.caption_type == 'kosmos2':
                        process = subprocess.Popen( ["/bin/bash", "run_kosmos_caption.sh", 
                                                     filter_dst, filter_dst],
                                                    cwd=osp.dirname(osp.abspath(__file__)),
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        stdout, stderr = process.communicate()
                        if process.returncode == 0:
                            logger.info(stdout.decode('utf-8'))
                        else:
                            logger.error(stderr.decode('utf-8'))
                            raise Exception
                            
                if settings.use_original_tags or settings.tag_using_wd14 or settings.enable_caption:
                    with jobContext(job_name="merge", file=file):
                        meta_file = osp.join(filter_dst,'meta_cap_dd.json')
                        if settings.use_original_tags:
                            merge_tag_extension = test_tag_extension(filter_dst, '.txt', tag_extension)
                            if merge_tag_extension is None:
                                logger.error("No tag file in source directory, unknow issue")
                                raise Exception
                        else:
                            merge_tag_extension = tag_extension
                        merge_arg = prepare_merge_parser(filter_dst, meta_file, merge_tag_extension)
                        merge_dd_tags_to_metadata.main(merge_arg)
                        if settings.enable_caption:
                            merge_arg = prepare_merge_parser(filter_dst, meta_file, caption_extension)
                            merge_captions_to_metadata.main(merge_arg)
                    
                    with jobContext(job_name="clean", file=file):
                        clean_args = prepare_clean_parser(meta_file,meta_file)
                        clean_captions_and_tags.main(clean_args)
                
                with jobContext(job_name="bucket", file=file):
                    # if json file doesn't exist, prepare buckets will run without it, so not making any change here
                    lat_file = osp.join(filter_dst,'meta_lat.json')
                    bucket_args = prepare_bucket_parser(filter_dst, meta_file, lat_file, sd_model_path,  
                                                        debug_dir=debug_dir,
                                                        resolution=settings.bucketing_resolution,
                                                        mixed_precision=settings.bucketing_mixed_precision,
                                                        batch_size=settings.bucketing_batch_szie,
                                                        flip_aug=settings.bucketing_flip_aug,
                                                        bucket_reso_steps=settings.bucketing_reso_steps
                                                        )
                    task = context.Process(target=prepare_buckets_latents.main, args=(bucket_args,))
                    task.start()
                    # cache_result = uploader.submit(cache_progress_watcher, settings.s3_cache_bucket_name, filter_dst, f"{osp.basename(target_dir)}_bucket", '.npz', recursive=True, interval=60*10)
                    task.join()
                    # Path.touch(osp.join(filter_dst, 'complete'))
                    # wait(cache_result)
                    
                if not settings.save_original_img:
                    remove_images_from_folder(filter_dst)
                    
                folders_to_compress = [filter_dst]
                if debug_dir:
                    folders_to_compress += [debug_dir]
                output_path = f"{target_dir}-{settings.complete_suffix}.tar.gz"
                
                uploader.submit(compress_and_upload, folders_to_compress, output_path, bucket_name)
                    
            except:
                # TODO kill uploader if exception occurs
                # if cache_result is not None:
                    # if not osp.isfile(osp.join(filter_dst, 'complete')):
                        # Path.touch(osp.join(filter_dst, 'complete'))
                    # wait(cache_result)
                logger.exception(f"Failed to process {file}")
        
    downloader.shutdown(wait=True)
    uploader.shutdown(wait=True)
    
if __name__ == '__main__':
    while True:
        main()
        time.sleep(settings.main_loop_interval)