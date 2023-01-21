#%%
import huggingface_hub
from huggingface_hub import ModelFilter
from huggingface_hub import HfApi

#%%
api = HfApi()

# %%
def model_exist(model_name):
    filter = ModelFilter(model_name =model_name)
    model = api.list_models(filter=filter)
    return len(model) > 0

def hf_login(token):
    huggingface_hub.login(token, add_to_git_credential=True)
