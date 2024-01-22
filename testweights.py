import cv2
import numpy as np

weights_file = "weights/yolov4-tiny-extra-batches-2_last.weights"
cfg_file = "cfg/yolov4-tiny-extra-batches-2.cfg"
image_file = "testimages/overpeckbike.png"


# Load class names
with open("Dataset/train/dataset.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO
net = cv2.dnn.readNet(weights_file, cfg_file)
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Load image
img = cv2.imread(image_file)
height, width, channels = img.shape

# Detect objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Get bounding boxes, confidences and class IDs
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Go through the detections remaining after nms and draw bounding box
for i in indices:
    if isinstance(i, (list, np.ndarray)) and len(i) > 0:
        i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    label = str(classes[class_ids[i]]) + " " + str(round(confidences[i], 2))
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# split image into 4x4 grid
grid_size = 4
cell_size_x = width // grid_size
cell_size_y = height // grid_size

for i in range(grid_size):
    for j in range(grid_size):
        start_x = i * cell_size_x
        start_y = j * cell_size_y
        end_x = start_x + cell_size_x
        end_y = start_y + cell_size_y
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1) 

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
