import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#WANDB_WATCH=all
#WANDB_SILENT=True

"""
import wandb
os.system("env WANDB_WATCH=all")
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Unaugmented",
    
    # track hyperparameters and run metadata
    config={

    "architecture": "DeiT",
    "dataset": "Types_bothsides_20_unaugmented_80_10_10",

    }
)

"""
print("~~~loading dataset~~~")
from datasets import load_dataset

# load the custom dataset
ds = load_dataset("imagefolder", data_dir="Datasets/coins_20_unaugmented_80_10_10")
print(ds)


import requests
import torch
from PIL import Image
import transformers
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import BeitForImageClassification, BeitImageProcessor
from transformers.modeling_outputs import SequenceClassifierOutput

# the model name
model_name = "microsoft/beit-base-patch16-224-pt22k"
# load the image processor
image_processor = BeitImageProcessor.from_pretrained(model_name)
# loading the pre-trained model
model =  BeitForImageClassification.from_pretrained(model_name)



import urllib.parse as parse


# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

def get_prediction(model, url_or_path):
  # load the image
  img = load_image(url_or_path)
  # preprocessing the image
  pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
  # perform inference
  output = model(pixel_values)
  # get the label id and return the class name
  return model.config.id2label[int(output.logits.softmax(dim=1).argmax())]

labels = ds["train"].features["label"]
print(labels)
print(labels.int2str(ds["train"][532]["label"]))


print("\n")
print("~~~preprocessing data~~~")

def transform(examples):
  # convert all images to RGB format, then preprocessing it
  # using our image processor
  inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
  # we also shouldn't forget about the labels
  inputs["labels"] = examples["label"]
  return inputs

# use the with_transform() method to apply the transform to the dataset on the fly during training
dataset = ds.with_transform(transform)
for item in dataset["train"]:
  #print(item)  
  print(item["pixel_values"].shape)
  print(item["labels"])
  break

# extract the labels for our dataset
labels = ds["train"].features["label"].names
print(labels)


def collate_fn(batch):
  return {
      "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
      "labels": torch.tensor([x["labels"] for x in batch]),
  }

from evaluate import load
import numpy as np

# load the accuracy and f1 metrics from the evaluate module
accuracy = load("accuracy")
f1 = load("f1")

def compute_metrics(eval_pred):
  # compute the accuracy and f1 scores & return them
  accuracy_score = accuracy.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)
  f1_score = f1.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids, average="macro")
  
  return {**accuracy_score, **f1_score}

print("~~~train model~~")
# load the Beit model
model = BeitForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./models/4", # output directory
  # output_dir="./vit-base-skin-cancer",
  per_device_train_batch_size=8, # batch size per device during training
  evaluation_strategy="steps",    # evaluation strategy to adopt during training
  num_train_epochs=25,             # total number of training epochs
  # fp16=True,                    # use mixed precision
  save_steps=500,                # number of update steps before saving checkpoint
  eval_steps=500,                # number of update steps before evaluating
  logging_steps=500,             # number of update steps before logging
  # save_steps=50,
  # eval_steps=50,
  # logging_steps=50,
  save_total_limit=2,             # limit the total amount of checkpoints on disk
  remove_unused_columns=False,    # remove unused columns from the dataset
  push_to_hub=False,              # do not push the model to the hub
  #report_to='tensorboard',        # report metrics to tensorboard
  load_best_model_at_end=True,    # load the best model at the end of training
  report_to="wandb",
  learning_rate=5e-05,
)



from transformers import Trainer

trainer = Trainer(
    model=model,                        # the instantiated 🤗 Transformers model to be trained
    args=training_args,                 # training arguments, defined above
    data_collator=collate_fn,           # the data collator that will be used for batching
    compute_metrics=compute_metrics,    # the metrics function that will be used for evaluation
    train_dataset=dataset["train"],     # training dataset
    eval_dataset=dataset["validation"], # evaluation dataset
    tokenizer=image_processor,          # the processor that will be used for preprocessing the images
)

# start training
trainer.train()
#trainer.save_model("path_to_save") 
trainer.evaluate(dataset["test"])

#trainer.save_model("./Modelle/DeiT")
"""
  learning_rate=5e-05,
  adam_beta1=0.9
  adam_beta2=0.999
  adam_epsilon=1e-08
"""
"""
def get_prediction_probs(model, url_or_path, num_classes=3):
    # load the image
    img = load_image(url_or_path)
    # preprocessing the image
    pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
    # perform inference
    output = model(pixel_values)
    # get the top k classes and probabilities
    probs, indices = torch.topk(output.logits.softmax(dim=1), k=num_classes)
    # get the class labels
    id2label = model.config.id2label
    classes = [id2label[idx.item()] for idx in indices[0]]
    # convert the probabilities to a list
    probs = probs.squeeze().tolist()
    # create a dictionary with the class names and probabilities
    results = dict(zip(classes, probs))
    return results

import time

for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        filepath = os.path.join(dirpath, filename)
        print("\n")
        print(filepath)
        print("\n")
        get_prediction("./models/checkpoint-4500", filepath)
        
        time.sleep(1)
        """

