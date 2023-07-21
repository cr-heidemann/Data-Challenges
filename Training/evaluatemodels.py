import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
from transformers import LevitModel, LevitImageProcessor, LevitForImageClassification, AutoImageProcessor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from datasets import load_dataset

# load the custom dataset
data_dir="Datasets/types_20_unaugmented_80_10_10/test/"
ds = load_dataset("imagefolder", data_dir="Datasets/types_20_unaugmented_80_10_10")
device = "cuda" if torch.cuda.is_available() else "cpu"
#Load Model from pretrained
#Specify TYPE and PATH
#model = TYPE.from_pretrained(PATH)
#e.g. model = BertModel.from_pretrained("./test/saved_model/")

model_name = "./models/1/checkpoint-4500/"
# load the image processor
image_processor = LevitImageProcessor.from_pretrained(model_name)
# loading the pre-trained model
model = LevitForImageClassification.from_pretrained(model_name)

# a function to load an image
def load_image(image_path):
    return Image.open(image_path)
    
def get_prediction_probs(model, url_or_path, num_classes=5):
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


# example 4
import time

for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        filepath = os.path.join(dirpath, filename)
        print("\n")
        print(filepath)
        print("\n")
        get_prediction_probs(model, filepath , num_classes=10)
        
        time.sleep(1)
"""        
for file in os.listdir(data_dir):
    path = os.path.join(data_dir, file)

"""

        
