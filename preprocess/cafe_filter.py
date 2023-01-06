#%%
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm


#%%
device = 0

class Aesthetic():
    
    def __init__(self, batch_size=3):
        pipe_aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic", device=device, batch_size=batch_size)
        spipe_style = pipeline("image-classification", "cafeai/cafe_style", device=device, batch_size=batch_size)
        pipe_waifu = pipeline("image-classification", "cafeai/cafe_waifu", device=device, batch_size=batch_size)
        self.pipes = [pipe_aesthetic, spipe_style, pipe_waifu]
        
    def calculate_aesthetic_score(self, images):
        results = {}
        with tqdm(total=len(self.pipes * len(images)), smoothing=0) as pbar:
            for pipe in self.pipes:
                for idx, result in enumerate(pipe(KeyDataset(images, 'image'), top_k=5)):
                    output = {}
                    for d in result:
                        output[d['label']] = d['score']
                    if idx in results:
                        results[idx] =  {**results[idx], **output}
                    else:
                        results[idx] = output
                    pbar.update(1)
        return results
# %%
# from glob import glob
# images = glob('/mnt/d/data/0/*.jpg')[0:2]
# images = [Image.open(i) for i in images]
# #%%
# a = Aesthetic()
# r = a.calculate_aesthetic_score(images)
# # %%
# a = Aesthetic()
# #%%
# img = '/mnt/c/Users/sheldon/Downloads/test/0_debug/robot_1671589798_446.jpg'
# a.calculate_aesthetic_score([Image.open(img)])
# %%
