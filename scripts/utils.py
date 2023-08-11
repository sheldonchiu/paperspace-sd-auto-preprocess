import os
import logging
from typing import Any, Callable, List, Union

logger = logging.getLogger(__name__)

class MinioClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = True) -> None:
        """
        Initializes a MinioClient object.

        Args:
            endpoint (str): The endpoint URL of the MinIO server.
            access_key (str): The access key for authentication.
            secret_key (str): The secret key for authentication.
            secure (bool, optional): Whether to use a secure connection (HTTPS) or not. Defaults to True.
        """
        from minio import Minio
        
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=secure
        )

    def list_objects(self, bucket_name: str) -> List[str]:
        """
        Lists the objects in a bucket.

        Args:
            bucket_name (str): The name of the bucket.

        Returns:
            list: A list of object names in the bucket.
        """
        try:
            response = [o.object_name for o in self.client.list_objects(bucket_name)]
        except Exception as e:
            print("Unable to list objects in bucket:", e)
            response = []
        return response

    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """
        Uploads a file to a bucket.

        Args:
            bucket_name (str): The name of the bucket.
            object_name (str): The name to give the uploaded object.
            file_path (str): The path to the file to upload.
        """
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print("File uploaded successfully")
        except Exception as e:
            print("Unable to upload file:", e)

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """
        Downloads a file from a bucket.

        Args:
            bucket_name (str): The name of the bucket.
            object_name (str): The name of the object to download.
            file_path (str): The path to save the downloaded file.
        """
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            print("File downloaded successfully")
        except Exception as e:
            print("Unable to download file:", e)

    def get_first_index_to_use(self, bucket_name: str) -> int:
        """
        Gets the first index to use.

        Args:
            bucket_name (str): The name of the bucket.

        Returns:
            int: The first index to use.
        """
        try:
            response = [
                int(o.object_name.replace(".tar.gz", ""))
                for o in self.client.list_objects(bucket_name)
                if "result" not in o.object_name
            ]
            response.sort()
            if len(response) == 0:
                return 0
            return response[-1] + 1
        except:
            logger.error("Unable to get first index to use")
            raise



class PaperspaceClient:
    def __init__(self, api_key: str) -> None:
        """
        Initializes the PaperspaceClient class.

        Args:
            api_key: The API key to authenticate requests.
        """
        from gradient import NotebooksClient
        from gradient import MachineTypesClient
        from gradient import ProjectsClient
    
        self.api_key = api_key
        self.notebooks_client = NotebooksClient(api_key)
        self.machineTypes_client = MachineTypesClient(api_key)
        self.projects_client = ProjectsClient(api_key)

    def find_available_gpu(self, priority: List[str]) -> Union[str, None]:
        """
        Finds the first available GPU from the given priority list.

        Args:
            priority: A list of GPU labels in priority order.

        Returns:
            The label of the first available GPU, or None if no available GPU is found.
        """
        VMs = [vm.label for vm in self.machineTypes_client.list()]
        for p in priority:
            if p in VMs:
                return p
        return None

    def get_notebook_detail(self, notebook_id: str) -> dict:
        """
        Retrieves the details of a notebook with the given ID.

        Args:
            notebook_id: The ID of the notebook.

        Returns:
            The details of the notebook as a dictionary.
        """
        return self.notebooks_client.get(id=notebook_id)

    def get_notobooks_by_project_id(self, project_id: str) -> List[dict]:
        """
        Retrieves all notebooks associated with the given project ID.

        Args:
            project_id: The ID of the project.

        Returns:
            A list of dictionaries containing the details of the notebooks.
        """
        notebooks = [n for n in self.notebooks_client.list(tags=[]) if n.project_handle==project_id]
        return notebooks