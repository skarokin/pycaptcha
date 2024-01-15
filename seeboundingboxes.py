# a tool to visualize bounding boxes on classes with matplotlib

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
from PIL import Image
import os

# get batch of images from training dataset of class x
class_dir = "Dataset/train/Stop sign"

images = []
bbs = []

for filename in os.listdir(class_dir):
    # get all jpg images
    if filename.endswith(".jpg") and filename.startswith("aug_"):
        filepath = os.path.join(class_dir, filename)
        image = Image.open(os.path.join(class_dir, filename))
        image = np.array(
            image,
            dtype=np.uint8
            )
        images.append(image)

        # extract bounding boxes and convert from normalized coordinates to pixel coordinates
        with open(filepath.replace(".jpg", ".txt"), "r") as f:
            bb = BoundingBoxesOnImage([
                BoundingBox(
                    x1=(float(line.split()[1]) - float(line.split()[3]) / 2) * image.shape[1],
                    y1=(float(line.split()[2]) - float(line.split()[4]) / 2) * image.shape[0],
                    x2=(float(line.split()[1]) + float(line.split()[3]) / 2) * image.shape[1],
                    y2=(float(line.split()[2]) + float(line.split()[4]) / 2) * image.shape[0],
                    label=line.split()[0]
                ) for line in f.readlines()
            ], shape=image.shape)
            bbs.append(bb)

# # augment images; if you want to see original images commnent these out and 
# # use images instead of images_aug and bb instead of bb_aug   
# seq = iaa.Sequential([
#     iaa.Fliplr(0.5),                        
#     iaa.Crop(percent=(0, 0.1)),             
#     iaa.Sometimes(
#         0.5,
#         iaa.GaussianBlur(sigma=(0, 3.0))    
#     ),
#     iaa.LinearContrast((0.75, 1.5)),         
#     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), 
#     iaa.Multiply((0.8, 1.2), per_channel=0.2), 
#     iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#         rotate=(-25, 25),
#         shear=(-8, 8)
#     ),
# ], random_order=True)   

# images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)

# remove bounding boxes that fall outside of image boundaries
for i in range(len(bbs)):
    bbs[i] = bbs[i].remove_out_of_image().clip_out_of_image()

for i in range(len(images)):

    image_bb = bbs[i].draw_on_image(images[i], size=2)

    plt.imshow(image_bb)
    plt.show()