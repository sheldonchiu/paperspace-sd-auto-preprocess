# python prepare_buckets_latents.py "/storage/text_image" "/storage/text_image/test.json" "/storage/text_image/meta_lat.json" "/tmp/sdxl_vae.safetensors" --recursive --batch_size=4 --max_data_loader_n_workers=2 --max_resolution="1024, 1024" --mixed_precision="bf16"
import ast
import toml
import os
import glob
import random

train_data_dir = "/tmp/0_filter"
output_dir = "/storage/train_output_2"
os.makedirs(output_dir, exist_ok=True)
config_dir = f"{output_dir}/config"
os.makedirs(config_dir, exist_ok=True)

# @title ## **4.1. LoRa: Low-Rank Adaptation Config**
# @markdown Kohya's `LoRA` renamed to `LoRA-LierLa` and Kohya's `LoCon` renamed to `LoRA-C3Lier`, read [official announcement](https://github.com/kohya-ss/sd-scripts/blob/849bc24d205a35fbe1b2a4063edd7172533c1c01/README.md#naming-of-lora).
network_category = "LoRA_C3Lier"  # @param ["LoRA_LierLa", "LoRA_C3Lier", "DyLoRA_LierLa", "DyLoRA_C3Lier", "LoCon", "LoHa", "IA3", "LoKR", "DyLoRA_Lycoris"]

# @markdown | network_category | network_dim | network_alpha | conv_dim | conv_alpha | unit |
# @markdown | :---: | :---: | :---: | :---: | :---: | :---: |
# @markdown | LoRA-LierLa | 32 | 1 | - | - | - |
# @markdown | LoCon/LoRA-C3Lier | 16 | 8 | 8 | 1 | - |
# @markdown | LoHa | 8 | 4 | 4 | 1 | - |
# @markdown | Other Category | ? | ? | ? | ? | - |

# @markdown Specify `network_args` to add `optional` training args, like for specifying each 25 block weight, read [this](https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E9%9A%8E%E5%B1%A4%E5%88%A5%E5%AD%A6%E7%BF%92%E7%8E%87)
network_args    = ""  # @param {'type':'string'}

# @markdown ### **Linear Layer Config**
# @markdown Used by all `network_category`. When in doubt, set `network_dim = network_alpha`
network_dim     = 16  # @param {'type':'number'}
network_alpha   = 8  # @param {'type':'number'}

# @markdown ### **Convolutional Layer Config**
# @markdown Only required if `network_category` is not `LoRA_LierLa`, as it involves training convolutional layers in addition to linear layers.
conv_dim        = 8  # @param {'type':'number'}
conv_alpha      = 1  # @param {'type':'number'}

# @markdown ### **DyLoRA Config**
# @markdown Only required if `network_category` is `DyLoRA_LierLa` and `DyLoRA_C3Lier`
unit = 8  # @param {'type':'number'}

if isinstance(network_args, str):
    network_args = network_args.strip()
    if network_args.startswith('[') and network_args.endswith(']'):
        try:
            network_args = ast.literal_eval(network_args)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing network_args: {e}\n")
            network_args = []
    elif len(network_args) > 0:
        print(f"WARNING! '{network_args}' is not a valid list! Put args like this: [\"args=1\", \"args=2\"]\n")
        network_args = []
    else:
        network_args = []
else:
    network_args = []

network_config = {
    "LoRA_LierLa": {
        "module": "networks.lora",
        "args"  : []
    },
    "LoRA_C3Lier": {
        "module": "networks.lora",
        "args"  : [
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "DyLoRA_LierLa": {
        "module": "networks.dylora",
        "args"  : [
            f"unit={unit}"
        ]
    },
    "DyLoRA_C3Lier": {
        "module": "networks.dylora",
        "args"  : [
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}",
            f"unit={unit}"
        ]
    },
    "LoCon": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=locon",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "LoHa": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=loha",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "IA3": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=ia3",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "LoKR": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=lokr",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "DyLoRA_Lycoris": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=dylora",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    }
}

network_module = network_config[network_category]["module"]
network_args.extend(network_config[network_category]["args"])

lora_config = {
    "additional_network_arguments": {
        "no_metadata"                     : False,
        "network_module"                  : network_module,
        "network_dim"                     : network_dim,
        "network_alpha"                   : network_alpha,
        "network_args"                    : network_args,
        "network_train_unet_only"         : True,
        "training_comment"                : None,
    },
}

print(toml.dumps(lora_config))

# @title ## **4.2. Optimizer Config**
# @markdown Use `Adafactor` optimizer. `RMSprop 8bit` or `Adagrad 8bit` may work. `AdamW 8bit` doesn't seem to work.
optimizer_type = "AdaFactor"  # @param ["AdamW", "AdamW8bit", "Lion8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation(DAdaptAdamPreprint)", "DAdaptAdaGrad", "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor"]
# @markdown Specify `optimizer_args` to add `additional` args for optimizer, e.g: `["weight_decay=0.6"]`
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ] # @param {'type':'string'}
# @markdown ### **Learning Rate Config**
# @markdown Different `optimizer_type` and `network_category` for some condition requires different learning rate. It's recommended to set `text_encoder_lr = 1/2 * unet_lr`
learning_rate = 1e-5  # @param {'type':'number'}
# @markdown ### **LR Scheduler Config**
# @markdown `lr_scheduler` provides several methods to adjust the learning rate based on the number of epochs.
lr_scheduler = "constant_with_warmup"  # @param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"] {allow-input: false}
lr_warmup_steps = 100  # @param {'type':'number'}
# @markdown Specify `lr_scheduler_num` with `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial`
lr_scheduler_num = 0  # @param {'type':'number'}

# if isinstance(optimizer_args, str):
#     optimizer_args = optimizer_args.strip()
#     if optimizer_args.startswith('[') and optimizer_args.endswith(']'):
#         try:
#             optimizer_args = ast.literal_eval(optimizer_args)
#         except (SyntaxError, ValueError) as e:
#             print(f"Error parsing optimizer_args: {e}\n")
#             optimizer_args = []
#     elif len(optimizer_args) > 0:
#         print(f"WARNING! '{optimizer_args}' is not a valid list! Put args like this: [\"args=1\", \"args=2\"]\n")
#         optimizer_args = []
#     else:
#         optimizer_args = []
# else:
#     optimizer_args = []

optimizer_config = {
    "optimizer_arguments": {
        "optimizer_type"          : optimizer_type,
        "learning_rate"           : learning_rate,
        "max_grad_norm"           : 0,
        "optimizer_args"          : optimizer_args,
        "lr_scheduler"            : lr_scheduler,
        "lr_warmup_steps"         : lr_warmup_steps,
        "lr_scheduler_num_cycles" : lr_scheduler_num if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power"      : lr_scheduler_num if lr_scheduler == "polynomial" else None,
        "lr_scheduler_type"       : None,
        "lr_scheduler_args"       : None,
    },
}

print(toml.dumps(optimizer_config))

# @markdown ### **Noise Control**
noise_control_type        = "noise_offset" #@param ["none", "noise_offset", "multires_noise"]
# @markdown #### **a. Noise Offset**
# @markdown Control and easily generating darker or light images by offset the noise when fine-tuning the model. Recommended value: `0.1`. Read [Diffusion With Offset Noise](https://www.crosslabs.org//blog/diffusion-with-offset-noise)
noise_offset_num          = 0.0357  # @param {type:"number"}
# @markdown **[Experimental]**
# @markdown Automatically adjusts the noise offset based on the absolute mean values of each channel in the latents when used with `--noise_offset`. Specify a value around 1/10 to the same magnitude as the `--noise_offset` for best results. Set `0` to disable.
adaptive_noise_scale      = 0.01 # @param {type:"number"}
# @markdown #### **b. Multires Noise**
# @markdown enable multires noise with this number of iterations (if enabled, around 6-10 is recommended)
multires_noise_iterations = 6 #@param {type:"slider", min:1, max:10, step:1}
multires_noise_discount = 0.3 #@param {type:"slider", min:0.1, max:1, step:0.1}
# @markdown ### **Custom Train Function**
# @markdown Gamma for reducing the weight of high-loss timesteps. Lower numbers have a stronger effect. The paper recommends `5`. Read the paper [here](https://arxiv.org/abs/2303.09556).
min_snr_gamma             = -1 #@param {type:"number"}

advanced_training_config = {
    "advanced_training_config": {
        "noise_offset"              : noise_offset_num if noise_control_type == "noise_offset" else None,
        "adaptive_noise_scale"      : adaptive_noise_scale if adaptive_noise_scale and noise_control_type == "noise_offset" else None,
        "multires_noise_iterations" : multires_noise_iterations if noise_control_type =="multires_noise" else None,
        "multires_noise_discount"   : multires_noise_discount if noise_control_type =="multires_noise" else None,
        "min_snr_gamma"             : min_snr_gamma if not min_snr_gamma == -1 else None,
    }
}

print(toml.dumps(advanced_training_config))

# @markdown ### **Project Config**
project_name                = "sdxl_lora"  # @param {type:"string"}
# @markdown Get your `wandb_api_key` [here](https://wandb.ai/settings) to logs with wandb.
wandb_api_key               = "" # @param {type:"string"}
in_json                     = f"{train_data_dir}/meta_lat.json"  # @param {type:"string"}
# @markdown ### **SDXL Config**
gradient_checkpointing      = True  # @param {type:"boolean"}
no_half_vae                 = True  # @param {type:"boolean"}
#@markdown Recommended parameter for SDXL training but if you enable it, `shuffle_caption` won't work
cache_text_encoder_outputs  = True  # @param {type:"boolean"}
#@markdown These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.
min_timestep                = 0 # @param {type:"number"}
max_timestep                = 1000 # @param {type:"number"}
# @markdown ### **Dataset Config**
num_repeats                 = 1  # @param {type:"number"}
resolution                  = 1024  # @param {type:"slider", min:512, max:1024, step:128}
keep_tokens                 = 0  # @param {type:"number"}
# caption_tag_dropout_rate
# @markdown ### **General Config**
num_epochs                  = 10  # @param {type:"number"}
train_batch_size            = 8  # @param {type:"number"}
mixed_precision             = "fp16"  # @param ["no","fp16","bf16"] {allow-input: false}
seed                        = -1  # @param {type:"number"}
optimization                = "scaled dot-product attention" # @param ["xformers", "scaled dot-product attention"]
# @markdown ### **Save Output Config**
save_precision              = "fp16"  # @param ["float", "fp16", "bf16"] {allow-input: false}
save_every_n_epochs         = 1  # @param {type:"number"}
# @markdown ### **Sample Prompt Config**
enable_sample               = True  # @param {type:"boolean"}
sampler                     = "euler_a"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
positive_prompt             = ""
negative_prompt             = ""
quality_prompt              = "Stable Diffusion XL"  # @param ["None", "Waifu Diffusion 1.5", "NovelAI", "AbyssOrangeMix", "Stable Diffusion XL"] {allow-input: false}
if quality_prompt          == "Waifu Diffusion 1.5":
    positive_prompt         = "(exceptional, best aesthetic, new, newest, best quality, masterpiece, extremely detailed, anime, waifu:1.2), "
    negative_prompt         = "lowres, ((bad anatomy)), ((bad hands)), missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts)), deleted, old, oldest, ((censored)), ((bad aesthetic)), (mosaic censoring, bar censor, blur censor), "
if quality_prompt          == "NovelAI":
    positive_prompt         = "masterpiece, best quality, "
    negative_prompt         = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, "
if quality_prompt         == "AbyssOrangeMix":
    positive_prompt         = "masterpiece, best quality, "
    negative_prompt         = "(worst quality, low quality:1.4), "
if quality_prompt          == "Stable Diffusion XL":
    negative_prompt         = "3d render, smooth, plastic, blurry, grainy, low-resolution, deep-fried, oversaturated"
custom_prompt               = "Huge robot with blue wing" # @param {type:"string"}
# @markdown Specify `prompt_from_caption` if you want to use caption as prompt instead. Will be chosen randomly.
prompt_from_caption         = "none"  # @param ["none", ".txt", ".caption"]
if prompt_from_caption != "none":
    custom_prompt           = ""
num_prompt                  = 2  # @param {type:"number"}
logging_dir                 = os.path.join(output_dir, "logs")


prompt_config = {
    "prompt": {
        "negative_prompt" : negative_prompt,
        "width"           : resolution,
        "height"          : resolution,
        "scale"           : 7,
        "sample_steps"    : 28,
        "subset"          : [],
    }
}

train_config = {
    "sdxl_arguments": {
        "cache_text_encoder_outputs" : cache_text_encoder_outputs,
        # "enable_bucket"              : True,
        "no_half_vae"                : True,
        # "cache_latents"              : True,
        # "cache_latents_to_disk"      : True,
        # "vae_batch_size"             : 2,
        "min_timestep"               : min_timestep,
        "max_timestep"               : max_timestep,
        "shuffle_caption"            : True if not cache_text_encoder_outputs else False,
    },
    "model_arguments": {
        "pretrained_model_name_or_path" : "/tmp/sd_xl_base_1.0.safetensors",
        "vae"                           : "/tmp/mecha_v2_e3-pruned.ckpt",
    },
    "dataset_arguments": {
        "debug_dataset"                 : False,
        "in_json"                       : in_json,
        "train_data_dir"                : train_data_dir,
        "dataset_repeats"               : num_repeats,
        "keep_tokens"                   : keep_tokens,
        "resolution"                    : str(resolution) + ',' + str(resolution),
        "color_aug"                     : False,
        "face_crop_aug_range"           : None,
        "token_warmup_min"              : 1,
        "token_warmup_step"             : 0,
    },
    "training_arguments": {
        "output_dir"                    : output_dir,
        "output_name"                   : project_name if project_name else "last",
        "save_precision"                : save_precision,
        "save_every_n_epochs"           : save_every_n_epochs,
        "save_n_epoch_ratio"            : None,
        "save_last_n_epochs"            : None,
        "save_state"                    : None,
        "save_last_n_epochs_state"      : None,
        "resume"                        : None,
        "train_batch_size"              : train_batch_size,
        "max_token_length"              : 225,
        "mem_eff_attn"                  : False,
        "sdpa"                          : True if optimization == "scaled dot-product attention" else False,
        "xformers"                      : True if optimization == "xformers" else False,
        "max_train_epochs"              : num_epochs,
        "max_data_loader_n_workers"     : 8,
        "persistent_data_loader_workers": True,
        "seed"                          : seed if seed > 0 else None,
        "gradient_checkpointing"        : gradient_checkpointing,
        "gradient_accumulation_steps"   : 1,
        "mixed_precision"               : mixed_precision,
    },
    "logging_arguments": {
        "log_with"          : "wandb" if wandb_api_key else "tensorboard",
        "log_tracker_name"  : project_name if wandb_api_key and not project_name == "last" else None,
        "logging_dir"       : logging_dir,
        "log_prefix"        : project_name if not wandb_api_key else None,
    },
    "sample_prompt_arguments": {
        "sample_every_n_steps"    : None,
        "sample_every_n_epochs"   : save_every_n_epochs if enable_sample else None,
        "sample_sampler"          : sampler,
    },
    "saving_arguments": {
        "save_model_as": "safetensors"
    },
}

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

def prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompt):
    if enable_sample:
        search_pattern = os.path.join(train_data_dir, '**/*' + prompt_from_caption)
        caption_files = glob.glob(search_pattern, recursive=True)

        if not caption_files:
            if not custom_prompt:
                custom_prompt = "masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
            new_prompt_config = prompt_config.copy()
            new_prompt_config['prompt']['subset'] = [
                {"prompt": positive_prompt + custom_prompt if positive_prompt else custom_prompt}
            ]
        else:
            selected_files = random.sample(caption_files, min(num_prompt, len(caption_files)))

            prompts = []
            for file in selected_files:
                with open(file, 'r') as f:
                    prompts.append(f.read().strip())

            new_prompt_config = prompt_config.copy()
            new_prompt_config['prompt']['subset'] = []

            for prompt in prompts:
                new_prompt = {
                    "prompt": positive_prompt + prompt if positive_prompt else prompt,
                }
                new_prompt_config['prompt']['subset'].append(new_prompt)

        return new_prompt_config
    else:
        return prompt_config

def eliminate_none_variable(config):
    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

    return config

try:
    train_config.update(optimizer_config)
except NameError:
    raise NameError("'optimizer_config' dictionary is missing. Please run  '4.1. Optimizer Config' cell.")

try:
    train_config.update(lora_config)
except NameError:
    raise NameError("'lora_config' dictionary is missing. Please run  '4.1. LoRa: Low-Rank Adaptation Config' cell.")

advanced_training_warning = False
try:
    train_config.update(advanced_training_config)
except NameError:
    advanced_training_warning = True
    pass

prompt_config = prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompt)

config_path         = os.path.join(config_dir, "config_file.toml")
prompt_path         = os.path.join(config_dir, "sample_prompt.toml")

config_str          = toml.dumps(eliminate_none_variable(train_config))
prompt_str          = toml.dumps(eliminate_none_variable(prompt_config))

write_file(config_path, config_str)
write_file(prompt_path, prompt_str)

print(config_str)

if advanced_training_warning:
    import textwrap
    error_message = "WARNING: This is not an error message, but the [advanced_training_config] dictionary is missing. Please run the '4.2. Advanced Training Config' cell if you intend to use it, or continue to the next step."
    wrapped_message = textwrap.fill(error_message, width=80)
    print('\033[38;2;204;102;102m' + wrapped_message + '\033[0m\n')
    pass

print(prompt_str)

#@markdown Check your config here if you want to edit something:
#@markdown - `sample_prompt` : /content/LoRA/config/sample_prompt.toml
#@markdown - `config_file` : /content/LoRA/config/config_file.toml


#@markdown You can import config from another session if you want.

sample_prompt   = f"{config_dir}/sample_prompt.toml" #@param {type:'string'}
config_file     = f"{config_dir}/config_file.toml" #@param {type:'string'}

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

def train(config):
    args = ""
    for k, v in config.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    return args

accelerate_conf = {
    "num_cpu_threads_per_process" : 8,
}

train_conf = {
    "sample_prompts"  : sample_prompt if os.path.exists(sample_prompt) else None,
    "config_file"     : config_file,
}

accelerate_args = train(accelerate_conf)
train_args = train(train_conf)

final_args = f"accelerate launch {accelerate_args} sdxl_train_network.py {train_args}"

print(final_args)