# pycaptcha
My custom CAPTCHA system leveraging YOLOv4 and Google Street View API!

# things-i-did
- Gathered images from Google's Open Images Dataset V4 with the help of this [repository](https://github.com/theAIGuysCode/OIDv4_ToolKit).
- Made a script (`preprocessing.py`) with **NumPy** and **imgaug** to produce random augmentations of ANY given dataset in YOLO format.
- Built a tool for visualizing bounding boxes (`seeboundingboxes.py`) with **Matplotlib**
- Installed Darknet, CUDA, and cuDNN to train YOLOv4-tiny locally (this took so long lol)
- Trained YOLOv4-tiny twice, once with augmented dataset (terrible), once with subset of original dataset (a little better but still sucks)

# things-to-do
- Fetch images from Google Street View API and run my model on it to get objects.
- Filter out bad images, store in database (probably Firestore; I still have a lot of free Google Cloud credits...)
- Upon test creation, display random image to the user and ask the user to select within a 3x3 grid where a certain object is...
  - This should be fun to implement! The current plan is to just do this all in the command line but I may make a GUI

# get-started
Run `pip install -r requirements.txt`, then run `captchatest.py`.
