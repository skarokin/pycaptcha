# pycaptcha
My custom CAPTCHA system leveraging YOLOv4 and Google Street View API!

# things-i-did
- Gathered images from Google's Open Images Dataset V4 with the help of this [repository](https://github.com/theAIGuysCode/OIDv4_ToolKit).
- Made a script (`preprocessing.py`) with **NumPy** and **imgaug** to produce random augmentations of ANY given dataset in YOLO format.
- Built a tool for visualizing bounding boxes (`seeboundingboxes.py`) with **Matplotlib**
- Installed Darknet, CUDA, and cuDNN to train YOLOv4-tiny locally (this took so long lol)
- Training YOLOv4-tiny (goal: <0.05 avg loss)
  - Full set of original images + a lot of augmentations (this sucked: 0.58 map, 0.5 avg loss)
  - Subset of original images with no augmentations (a little better: 0.51 map, 0.36 avg loss)
  - Full set of original images + a little augmentation (hmmm: 0.69 map, 0.44 avg loss)
  - Full set of original images + 30% of images with slight augmentations (0.71 map, 0.45 avg loss)
  - Transfer learning w/ first 29 layers...
    - Full set of original images + 30% of images with slight augmentation (i'm getting there! 0.79 map, 0.24 avg loss)
    - Doubled batches, full set of original images + 30% of images with slight augmentations (0.79 map, 0.17 avg loss)
    - Doubled batches, full set of original images + 60% of images with slight augmentations (0.75 map, 0.17 avg loss)
      - Fine-tuned from best weights of above... (0.85 map, 0.25 avg loss)

End result of training:
- Outperformed YOLOv4-tiny using my custom dataset (COCO-trained model has the same classes; my model has better mAP on these classes compared to COCO-trained)
- But in the end I decided to just use COCO-trained YOLOv4 since it's better than tiny anyway lol 

# things-to-do
- Fetch images from Google Street View API and run my model on it to get objects.
- Filter out bad images, store in database (probably Firestore; I still have a lot of free Google Cloud credits...)
- Upon test creation, display random image to the user and ask the user to select within a 3x3 grid where a certain object is...
  - This should be fun to implement! The current plan is to just do this all in the command line but I may make a GUI

# get-started
Run `pip install -r requirements.txt`, then run `captchatest.py`.
