# pycaptcha
My custom CAPTCHA system leveraging YOLOv4 and Google Street View API!

# things-i-did
- Gathered images from Google's Open Images Dataset V4 with the help of this [repository](https://github.com/theAIGuysCode/OIDv4_ToolKit).
- Developed a CLI tool (`preprocessing.py`) with **imgaug** and **NumPy** to produce random augmentations of any given dataset in YOLO format.
- Built a tool for visualizing bounding boxes (`seeboundingboxes.py`) with **Matplotlib**

# things-to-do
- Train YOLOv4 on my new dataset (I have an RTX 4070 should be fine :D).
- Fetch images from Google Street View API and run my model on it, getting outputs.
- Filter out bad images, store in database, fetch to run tests!.

# get-started
Run `pip install -r requirements.txt`, then run `captchatest.py`.
