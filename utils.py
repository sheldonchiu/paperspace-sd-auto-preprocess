import os
import logging
from minio import Minio
from gradient import NotebooksClient
from gradient import MachineTypesClient
from gradient import ProjectsClient

logger = logging.getLogger(__name__)

api_key = os.environ['PAPERSPACE_API_KEY']
priority = ['Free-A4000','Free-RTX5000', 'Free-P5000']

s3 = Minio(
    os.environ['S3_HOST_URL'],
    access_key=os.environ['S3_ACCESS_KEY'],
    secret_key=os.environ['S3_SECRET_KEY'],
    secure=False
)

notebooks_client = NotebooksClient(api_key)
machineTypes_client = MachineTypesClient(api_key)
projects_client = ProjectsClient(api_key)

def get_list_of_files(bucketName):
    try:
        response = [o.object_name for o in s3.list_objects(bucketName)]
    except KeyError:
        response = []
    except:
        logger.error("Unable to list objects in bucket, please check the s3 storage")
    return response

def find_available_gpu():
    VMs = [vm.label for vm in machineTypes_client.list()]
    for p in priority:
        if p in VMs:
            return p
    return None

def get_notebook_detail(notebook_id):
    return notebooks_client.get(id=notebook_id)