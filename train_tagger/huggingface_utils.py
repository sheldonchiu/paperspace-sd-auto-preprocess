#%%
from huggingface_hub import ModelFilter
from huggingface_hub import HfApi

#%%
api = HfApi()

# %%
def model_exist(model_name):
    filter = ModelFilter(model_name =model_name)
    model = api.list_models(filter=filter)
    return len(model) > 0
