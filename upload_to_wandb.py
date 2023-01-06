#%%
import wandb

run = wandb.init(project="stable-diffusion-mecha")
artifact = wandb.Artifact('danbooru-mech-v1', type='dataset')
artifact.add_dir('output') # Adds multiple files to artifact
run.log_artifact(artifact) # Creates `animals:v0`