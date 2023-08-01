# %%
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time
import pytz
import datetime
import random
from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.INFO)


# %%
load_dotenv()
from utils import *
# %%
project_id = os.environ['PROJECT_ID']
bucket_name = os.environ['S3_BUCKET_NAME']
complete_suffix = os.environ['COMPLETE_SUFFIX']
no_work_sleep_interval = 60 * 5
exception_sleep_interval = 60 * 5
gpu_wait_interval = 60
running_sleep_interval = 60 * 10
notebook_base_url = "https://{}.{}.paperspacegradient.com?token={}"
extra_env = [e for e in os.environ.keys() if 'ENV_' in e]

options = Options()
# if(headless):
#     options.add_argument('--headless')
# os_type="linux-aarch64"

driver = webdriver.Firefox(
    executable_path=GeckoDriverManager().install(), options=options)
driver.set_window_size(1920, 1080)
while True:
    try:
        # get list of pending job
        files_to_process = []
        files = get_list_of_files(bucket_name)
        for f in files:
            filename = f[:f.find('.')]
            if complete_suffix not in f and f'{filename}-{complete_suffix}.tar.gz' not in files:
                files_to_process.append(f)

        if len(files_to_process) > 0:
            logging.info('Pending job found, start processing')
            notebooks = get_notobooks_by_project_id(project_id)
            running = False
            for notebook_status in notebooks:
                if notebook_status.state == "Running":
                    running = True
                    break
                else:
                    notebooks_client.delete(id=notebook_status.id)
            if running:
                logging.info(f"Notebook is already running...")
                # sleep until notebook ends if it is running
                notebook_start_time = notebook_status.dt_started.astimezone(
                    pytz.utc)
                notebook_end_time = notebook_start_time + \
                    datetime.timedelta(hours=notebook_status.shutdown_timeout)
                notebook_url = notebook_base_url.format(
                    notebook_status.id, notebook_status.cluster_id, notebook_status.token)
                # handle case when notebook is not ready yet
                driver.get(notebook_url)
                try:
                    element = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((
                        By.XPATH, '//div[@title="Start a new terminal session"]')))
                    element.click()
                except:
                    logging.info(
                        "start terminal button not found, maybe already running, continuing...")
                time.sleep(5)
                while True:
                    try:
                        # terminal = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(By.CLASS_NAME, "xterm-helper-textarea"))
                        terminal = driver.find_element_by_class_name(
                            "xterm-helper-textarea")
                        terminal.send_keys("ls \n")
                        time.sleep(running_sleep_interval)
                        # backup method to terminate
                        if (notebook_end_time - datetime.datetime.now().astimezone(pytz.utc)).total_seconds() <= 0:
                            break
                    except:
                        logging.exception(
                            "Failed to enter command into terminal")
                        break
                continue
            else:
                gpu = find_available_gpu()
                if gpu is None:
                    while True:
                        logging.info("No matching GPU available")
                        time.sleep(gpu_wait_interval)
                        gpu = find_available_gpu()
                        if gpu is not None:
                            break
                try:
                    # required environment variables
                    environment = {
                        "S3_HOST_URL": os.environ['S3_HOST_URL_EXT'],
                        "S3_ACCESS_KEY": os.environ['S3_ACCESS_KEY'],
                        "S3_SECRET_KEY": os.environ['S3_SECRET_KEY'],
                        "S3_BUCKET_NAME": os.environ['S3_BUCKET_NAME'],
                        "COMPLETE_SUFFIX": complete_suffix,
                        "RUN_SCRIPT": "kosmos2,preprocess",
                    }
                    environment.update({e.replace('ENV_', ''): os.environ[e] for e in extra_env})
                    # optional environment variables
                    discord_webhook_url = os.environ.get(
                        'DISCORD_WEBHOOK_URL', None)
                    if discord_webhook_url is not None:
                        environment['DISCORD_WEBHOOK_URL'] = discord_webhook_url

                    notebook_id = notebooks_client.create(
                        machine_type=gpu,
                        container='paperspace/gradient-base:pt112-tf29-jax0317-py39-20230125',
                        project_id=project_id,
                        name='auto',
                        command="bash entry.sh >> /tmp/run.log & jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True",
                        shutdown_timeout="6",
                        workspace="https://github.com/sheldonchiu/Ultimate-Paperspace-Template.git",
                        environment=environment
                    )
                    logging.info("Notebook successfully started")
                    # wait for the notebook to start and check in next iteration
                    time.sleep(no_work_sleep_interval)
                    continue
                except:
                    logging.exception(f"Failed to start notebook")
                    time.sleep(exception_sleep_interval)
                    continue
        else:
            # if no pending work, then sleep
            logging.info("No job was find, sleeping...")
            time.sleep(no_work_sleep_interval)

    except KeyboardInterrupt:
        logging.info("Interrupted")
        break
    except:
        logging.exception("Unknown error")
        time.sleep(5)

driver.close()
