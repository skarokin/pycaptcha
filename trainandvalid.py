# this script takes all images from dataset and 
# lists the image paths in a text file. 70% are training and 30% are validation
# this is to configure darknet to train on the dataset

import os
import random

dataset_dir = "Dataset/train"

# walk through dataset directory
def walk_dataset():
    for class_dir in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, class_dir)):
            print(f"Processing class {class_dir}...")
            get_image_paths(os.path.join(dataset_dir, class_dir))

# get all image paths, shuffle randomly, split into 70% training and 30% validation
# finally, write to "dataset_train.txt" and "dataset_valid.txt"        
def get_image_paths(class_dir):
    image_paths = []
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpg"):
            image_paths.append(os.path.join(class_dir, filename))
    random.shuffle(image_paths)
    split = int(0.7 * len(image_paths))
    train_paths = image_paths[:split]
    valid_paths = image_paths[split:]
    with open("Dataset/train/dataset_train.txt", "a") as f:
        for path in train_paths:
            print("wrote path of" + path + "to dataset_train.txt")
            # ../../pycaptcha/Dataset/train/class_dir/filename.jpg
            f.write("../../pycaptcha/" + path.replace("\\", "/") + "\n")
    with open("Dataset/train/dataset_valid.txt", "a") as f:
        for path in valid_paths:
            print("wrote path of" + path + "to dataset_valid.txt")
            f.write("../../pycaptcha/" + path.replace("\\", "/") + "\n")
    print("done processing" + class_dir)

walk_dataset()