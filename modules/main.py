from get_inferences import infer_image
from draw_boxes import draw_boxes
import pandas as pd
from streetview import get_streetview
from io import BytesIO
import numpy as np
from PIL import Image
import sys

# load class names from local
with open("Dataset/train/datasetpretrained.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# all this shit below is for when street view api works guhhhhhhh
# extract an image from street view
# image_obj = get_streetview(
#     pano_id="KAHiCou43QQzBb6w6",
#     api_key='AIzaSyAyNR8NrRqotWQAXlMNd8oZjdOnohQo7H0',
# )
# print(image_obj)
# # Convert the JpegImageFile to a bytes-like object
# byte_stream = BytesIO()
# image_obj.save(byte_stream, format='JPEG')
# image_data = byte_stream.getvalue()  # Now image_data is a bytes-like object

# # Convert the bytes-like object to a numpy array
# image = Image.open(BytesIO(image_data))
# image = np.array(image)[:,:,::-1].copy()
    
image_path = "testimages/palpktrafficlight.png"

# get_inferences returns a tensor and applies non-max suppression
tensor = infer_image(image_path, classes)

# turn tensor into pandas dataframe for easier processing
# everything is int except for confidence
df = pd.DataFrame(tensor, columns=['x', 'y', 'w', 'h', 'class_id', 'confidence'])
for col in df.columns:
    if col != 'confidence':
        df[col] = df[col].astype(int)
print(df)

# when initializing captcha test, go through dataframe and identify class with highest avg confidence
# group by class_id, take the mean confidence of each class_id, and choose class with the highest mean confidence
if not df.empty:
    avg_confidence = df.groupby('class_id')['confidence'].mean()
    class_to_test = avg_confidence.idxmax()
    print(f"class with highest avg confidence: {classes[class_to_test]}")
    if not all(df[df['class_id'] == class_to_test]['confidence'] > 0.50):
        print("Unconfident about some detections; try a new test")
        sys.exit()
else:
    print("No inferences made; try a new test")

# split image into selectable 5x5 grid

# prompt user to click among a 5x5 grid all instances of the selected class  

# draw boxes based on tensor (this will later be changed to be the actual captcha test)
draw_boxes(image_path, classes, tensor)