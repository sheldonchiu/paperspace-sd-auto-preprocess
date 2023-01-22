#%%
#docker run -it --rm -v $PWD:/app --name "train" --ipc=host  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime /bin/bash
# pip install -q transformers datasets scikit-learn 
# git config --global credential.helper store
#
import os
import numpy as np
from datasets import load_dataset 
import torch
from torch import nn
import itertools
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EvalPrediction,
)
from transformers.trainer_utils import get_last_checkpoint
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    ColorJitter,
                                    Resize, 
                                    ToTensor)
import torchvision.transforms.functional as F
import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# model_checkpoint = "/beit-base-patch16-384" # pre-trained model from which to fine-tune
# model_checkpoint = "microsoft/beit-base-patch16-384"
# batch_size = 32 # batch size for training and evaluation
# max_train_samples = None
# max_eval_samples = None
# seed = 42
# cache_dir = None

class SquarePad:
    def __call__(self, image):
        s = image.size()
        max_wh = np.max(s[-1], s[-2])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

#%%
def main(model_checkpoint, class_file, data_dir,
         batch_size=32, 
         max_train_samples=None, 
         max_eval_samples=None, 
         seed=42, cache_dir=None,
         eval_metric="f1", eval_threshold=0.6,
         output_dir='/tmp/mecha-tagger', log_to=None,
         lr_scheduler_type='constant',
         warmup_ratio=0.05,
         learning_rate=2e-5,
         num_train_epochs=50,
         weight_decay=0.3,
         gradient_accumulation_steps=1,
         gradient_checkpointing=True,
         hub_id=None):
    
    set_seed(seed)
    with open(class_file, "r") as f:
        class_list = [c.strip() for c in f.readlines()]
    num_of_class = len(class_list)
    print(f"number of class: {num_of_class}")

    dataset = load_dataset("imagefolder",data_dir=data_dir)
    splits = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
    train_ds = splits['train']
    val_ds = splits['test']

    all_labels = [l for l in list(itertools.chain(*train_ds['labels'])) if l in class_list]
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array(class_list),y=all_labels)
    
    id2label = {id:label for id, label in enumerate(class_list)}
    label2id = {label:id for id,label in id2label.items()}


    config = AutoConfig.from_pretrained(
            model_checkpoint,
            num_labels=num_of_class,
            label2id=label2id,
            id2label=id2label,
            finetuning_task="image-classification",
            cache_dir=cache_dir,
            problem_type="multi_label_classification"
    )
    model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_checkpoint,
            cache_dir=cache_dir)

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in feature_extractor.size:
        size = feature_extractor.size["shortest_edge"]
    else:
        size = (feature_extractor.size["height"], feature_extractor.size["width"])
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    _train_transforms = Compose(
        [
            # RandomRotation((0,360)),
            ColorJitter(brightness=0.5,hue=0.5),
            SquarePad(),
            # RandomResizedCrop(size),
            Resize(size),
            # CenterCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        labels = []
        for example in examples['labels']:
            one_hot = [0.0] * num_of_class
            for label in example:
                if label in label2id:
                    one_hot[label2id[label]] = 1.0
            labels.append(one_hot)
        examples['labels'] = labels
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        labels = []
        for example in examples['labels']:
            one_hot = [0.0] * num_of_class
            for label in example:
                if label in label2id:
                    one_hot[label2id[label]] = 1.0
            labels.append(one_hot)
        examples['labels'] = labels
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.stack([torch.tensor(example["labels"]) for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Set the transforms
    train_ds.set_transform(train_transforms)
    if max_train_samples:
        train_ds.shuffle(seed=seed).select(range(max_train_samples))
    val_ds.set_transform(val_transforms)
    if max_eval_samples:
        val_ds.shuffle(seed=seed).select(range(max_eval_samples))

#%%
    # os.environ["WANDB_DISABLED"] = "true"

    training_args = TrainingArguments(
        output_dir,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # eval_steps=1,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        # load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
        logging_dir='/tf_logs',
        remove_unused_columns=False,
        report_to=log_to,
        optim="adamw_torch",
        logging_steps = 100,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # auto_find_batch_size=True,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=4,
        push_to_hub= hub_id is not None,
        hub_model_id=hub_id,
        hub_strategy="checkpoint"
    )
     
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = os.path.join(training_args.output_dir, "last-checkpoint")

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids,
            threshold=eval_threshold)
        return result

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = nn.BCEWithLogitsLoss(weight=torch.tensor(class_weights).to(device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result  = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

# os.system("shutdown")

if __name__ == "__main__":
    main("microsoft/beit-base-patch16-384",
         "/tmp/data/class.txt",
         "/tmp/data/data_resized",
        log_to=['wandb',], batch_size=96, gradient_accumulation_steps=2)