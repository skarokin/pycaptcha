from get_inferences import infer_image
from draw_boxes import draw_boxes
import pandas as pd

# Load class names
with open("Dataset/train/datasetpretrained.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

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

# when initializing captcha test, go through dataframe and choose class with highest avg confidence
# group by class_id, take the mean of confidence, and choose the class with the highest mean confidence
avg_confidence = df.groupby('class_id')['confidence'].mean()
print(f"class with highest avg confidence: {classes[avg_confidence.idxmax()]}")

# split image into selectable 5x5 grid

# prompt user to click among a 5x5 grid all instances of the class with highest avg confidence   

# draw boxes based on tensor (this will later be changed to be the actual captcha test)
draw_boxes(image_path, classes, tensor)