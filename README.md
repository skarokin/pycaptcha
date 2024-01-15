# pycaptcha
My custom CAPTCHA system leveraging YOLOv4 and Google Street View API!

# things-i-did
- Gathered images from Google's Open Images Dataset V4 with the help of this [repository](https://github.com/theAIGuysCode/OIDv4_ToolKit).
- Made a script (`preprocessing.py`) with **imgaug** and **NumPy** to produce random augmentations of ANY given dataset in YOLO format.
- Built a tool for visualizing bounding boxes (`seeboundingboxes.py`) with **Matplotlib**

# things-to-do
- Train YOLOv4 on my new dataset (I have an RTX 4070 I should be fine :D).
- Fetch images from Google Street View API and run my model on it to get objects.
- Filter out bad images, store in database (probably Firestore; I still have a lot of free Google Cloud credits...)
- Upon test creation, display random image to the user and ask the user to select within a 3x3 grid where a certain object is...
  - This should be fun to implement! The current plan is to just do this all in the command line but I may make a GUI

# get-started
Run `pip install -r requirements.txt`, then run `captchatest.py`.
