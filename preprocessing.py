# a little tool for augmenting images and bounding boxes in YOLO format
# walks through a directory of classes, loads images and bounding boxes, and performs random augmentations
# augmentations are saved to the same directory with a different name
# file structure should be as such:
# |-- Dataset
# |    |-- train
# |    |    |-- class1
# |    |    |    |-- image1.jpg
# |    |    |    |-- image1.txt
# |    |    |-- class2
# |    |-- test
# ...

# NOTE: this tool assumes none of your images begin with "aug_" and all images are already labeled in YOLO format

import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image
import os
import random

# augmentation pipeline, see imgaug documentation for more info (and to add more if you want)
seq = iaa.Sequential([
    iaa.Fliplr(0.5),                        
    iaa.Crop(percent=(0, 0.1)),             
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 3.0))    
    ),
    iaa.LinearContrast((0.75, 1.5)),         
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), 
    iaa.Multiply((0.8, 1.2), per_channel=0.2), 
], random_order=True)                

# replace with your own dataset directory
dataset_dir = "Dataset/train"

# perform process_images on all classes in dataset
def walk_dataset(dataset_dir):
    images_current_class = []
    bbs_current_class = []

    for class_dir in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, class_dir)):
            print(f"Processing class {class_dir}...")
            process_images(os.path.join(dataset_dir, class_dir), images_current_class, bbs_current_class)
            images_current_class.clear()    
            bbs_current_class.clear()

# process all images in a class directory
# opens the image and bounding boxes, converts to numpy array, 
# then performs random augmentations and saves to the same directory with different name
def process_images(class_dir, images, bbs):

    for filename in os.listdir(class_dir):
        # get all jpg images
        if filename.endswith(".jpg"):
            filepath = os.path.join(class_dir, filename)
            image = Image.open(os.path.join(class_dir, filename))
            # this step is crucial because imgaug only works on numpy arrays
            image = np.array(
                    image,
                    dtype=np.uint8
            )
            images.append(image)

            # extract bounding boxes and convert to pixel coordinates 
            with open(filepath.replace(".jpg", ".txt"), "r") as f:
                bb = BoundingBoxesOnImage([
                    BoundingBox(
                        x1=(float(line.split()[1]) - float(line.split()[3]) / 2) * image.shape[1],
                        y1=(float(line.split()[2]) - float(line.split()[4]) / 2) * image.shape[0],
                        x2=(float(line.split()[1]) + float(line.split()[3]) / 2) * image.shape[1],
                        y2=(float(line.split()[2]) + float(line.split()[4]) / 2) * image.shape[0],
                        label=line.split()[0],
                    ) for line in f.readlines()
                ], shape=image.shape)
                bbs.append(bb)

    print(f"Images and bounding boxes loaded for {class_dir}...")

    # perform random augmentations for 30% of images
    augmentation_probability = 0.3
    images_aug, bbs_aug = augment(images, bbs, augmentation_probability)

    print(f"Augmentations complete for {class_dir}...")

    # finally, save augmented images and bounding boxes to directory with new name
    save_to_directory(class_dir, images_aug, bbs_aug)

    print(f"Successfully saved augmentations to {class_dir}!")

# perform random augmentations with predefined pipeline, must use pixel coordinates
# imgaug automatically moves bounding boxes for me :D
def augment(images, bbs, augmentation_probability):
    images_aug = []
    bbs_aug = []

    # random chance to perform augmentations, using zip to traverse lists in parallel
    # only append images that have been augmented
    for image, bb in zip(images, bbs):
        if random.random() < augmentation_probability:
            image_aug, bb_aug = seq(image=image, bounding_boxes=bb)
        images_aug.append(image_aug)
        bbs_aug.append(bb_aug)

    # remove out of bounds and clipped bounding boxes
    for i in range(len(bbs_aug)):
        bbs_aug[i] = bbs_aug[i].remove_out_of_image().clip_out_of_image()

    return images_aug, bbs_aug

# save to directory with new name
def save_to_directory(class_dir, images_aug, bbs_aug):
    for i in range(len(images_aug)):
        # save images by index
        image_aug = Image.fromarray(images_aug[i])
        image_aug.save(os.path.join(class_dir, f"aug_{i}.jpg"))

        # save each bounding box of each image to a new line in a text file
        # ensuring that the bounding boxes are converted to normalized coordinates
        with open(os.path.join(class_dir, f"aug_{i}.txt"), "w") as f:
            for bb in bbs_aug[i].bounding_boxes:
                normalize(bb, images_aug[i].shape[1], images_aug[i].shape[0])
                f.write(f"{bb.label} {bb.x1} {bb.y1} {bb.x2} {bb.y2}\n")

# pixel to normalized coordinates
# xcenter = (x1 + w/2) / image_width
# ycenter = (y1 + h/2) / image_height
# width = w / image_width
# height = h / image_height
def normalize(bb, image_width, image_height):
    w = bb.x2 - bb.x1
    h = bb.y2 - bb.y1

    bb.x1 = (bb.x1 + w / 2) / image_width
    bb.y1 = (bb.y1 + h / 2) / image_height
    bb.x2 = w / image_width
    bb.y2 = h / image_height

walk_dataset(dataset_dir)