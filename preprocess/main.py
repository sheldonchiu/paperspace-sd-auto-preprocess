import os
import sys
import os.path as osp

# will add path to kohya trainer in settings
import logging
from utils import *
logger = logging.getLogger()
setup_logger(logger)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import active_children
import signal
import settings

import file_filter
import make_captions
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
    if not settings.use_original_tags and not settings.tag_using_wd14:
        logger.error("cannot disable wd14 tagger and not use original tags")
        sys.exit(-1)
        
    signal.signal(signal.SIGTERM, sigterm_handler)
    downloader = ProcessPoolExecutor(max_workers=1)
    uploader = ProcessPoolExecutor(max_workers=1)
    # fork will cause tf cudu init error, unknown reason
    context = multiprocessing.get_context('spawn')
    bucket_name = os.environ['S3_BUCKET_NAME']
    files = get_list_of_files(bucket_name)
    #0.tar.gz -> 0-result.tar.gz
    files_to_process = []
    tag_extension = '.tag'
    caption_extension = '.caption'
    for f in files:
        filename = f[:f.find('.')]
        if 'result' not in f and f'{filename}-result.tar.gz' not in files:
            files_to_process.append(f)
            
    results = downloader.map(download, [(bucket_name, f, osp.join(settings.data_download_path, osp.basename(f))) for f in files_to_process])
    for file in files_to_process:
        local_path = osp.join(settings.data_download_path, osp.basename(file))
        if next(results) and (target_dir := extract(local_path)):
            try:
                filter_dst = f"{target_dir}_filter" if settings.enable_filter else target_dir
                debug_dir = f"{target_dir}_debug" if settings.save_img_for_debug else None
                
                if debug_dir is not None:
                    os.makedirs(debug_dir, exist_ok=True)
                    
                if settings.tag_using_wd14:
                    # create tag using wd14 and storing with .tag extension in the same directory
                    logger.info(f"Start tagging for {file}")
                    wd_args = prepare_wd_parser(target_dir, batch_size=settings.wd14_batch_size, caption_extention=tag_extension)
                    task = context.Process(target=tag_images_by_wd14_tagger.main, args=(wd_args,))
                    task.start()
                    task.join()
                    logger.info(f"Finish tagging for {file}")
                    
                if settings.caption_using_blip:
                    logger.info(f"Start captioning for {file}")
                    blip_args = prepare_caption_parser(target_dir, batch_size=settings.blip_batch_size, caption_extention=caption_extension)
                    task = context.Process(target=make_captions.main, args=(blip_args,))
                    task.start()
                    task.join()
                    logger.info(f"Finish captioning for {file}")
                
                if settings.enable_filter:
                    # use tag created by wd14 and filter, save symbolic links in folder {train_dir}_filter
                    logger.info(f"Start filter for {file}")
                    # since data dir has changed, need to update target_dir
                    task = context.Process(target=file_filter.main, args=(target_dir, filter_dst,tag_extension, caption_extension, settings.filter_using_cafe_aesthetic,debug_dir))
                    task.start()
                    task.join()
                    logger.info(f"Finish filter for {file}")
                
                logger.info(f"Start metadata merging for {file}")
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
                if settings.caption_using_blip:
                    merge_arg = prepare_merge_parser(filter_dst, meta_file, caption_extension)
                    merge_captions_to_metadata.main(merge_arg)
                logger.info(f"Finish metadata merging for {file}")
                
                logger.info(f"Start cleaning metadata for {file}")
                clean_args = prepare_clean_parser(meta_file,meta_file)
                clean_captions_and_tags.main(clean_args)
                logger.info(f"Finish cleaning metadata for {file}")
                
                logger.info(f"Start bucketing for {file}")
                lat_file = osp.join(filter_dst,'meta_lat.json')
                if settings.vae_model_url:
                    sd_model_path = download_model(osp.join(settings.model_path, 'stable-diffusion'), settings.vae_model_url)
                else:
                    sd_model_path = settings.vae_model_hub
                
                if settings.vae_model_url_2:
                    sd_model_path_2 = download_model(osp.join(settings.model_path, 'stable-diffusion'), settings.vae_model_url_2)
                else:
                    sd_model_path_2 = settings.vae_model_hub_2
                
                bucket_args = prepare_bucket_parser(filter_dst, meta_file, lat_file, sd_model_path, 
                                                    osp.join(settings.model_path, 'upscaler'), 
                                                    debug_dir=debug_dir,model_name_or_path_v2=sd_model_path_2,
                                                    upscale_outscale=settings.upscale_outscale
                                                    )
                task = context.Process(target=prepare_buckets_latents.main, args=(bucket_args,))
                task.start()
                task.join()
                logger.info(f"Finish bucketing for {file}")
                
                folders_to_compress = [filter_dst]
                if debug_dir:
                    folders_to_compress += [debug_dir]
                output_path = f"{target_dir}-result.tar.gz"
                
                uploader.submit(compress_and_upload, folders_to_compress, output_path, bucket_name)
                    
            except:
                logger.exception(f"Failed to process {file}")
        
    downloader.shutdown(wait=True)
    
if __name__ == '__main__':
    main()