#%%
import huggingface_hub
from huggingface_hub import ModelFilter
from huggingface_hub import HfApi
from datetime import datetime, timezone
import pytz
#%%
api = HfApi()

# %%
def model_exist(model_name):
    filter = ModelFilter(model_name =model_name)
    model = api.list_models(filter=filter)
    return len(model) > 0

def model_last_uptime_time(model_name):
    filter = ModelFilter(model_name =model_name)
    model = api.list_models(filter=filter)[0]
    return datetime.strptime(model.lastModified, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)

def hf_login(token):
    huggingface_hub.login(token, add_to_git_credential=True)
