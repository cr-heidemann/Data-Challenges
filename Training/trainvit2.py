import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset

train_ds = load_dataset('coins4/train/')
valid_ds = load_dataset('coins4/valid/')
test_ds = load_dataset('coins4/test/')

id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
print(id2label)

